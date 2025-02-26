#include "scene.h"

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/norm.hpp>

#include <algorithm>
#include <filesystem>

#include <iostream>

namespace fs = std::filesystem;

namespace oka
{

uint32_t Scene::createMesh(const std::vector<Vertex>& vb, const std::vector<uint32_t>& ib)
{
    std::scoped_lock lock(mMeshMutex);

    Mesh* mesh = nullptr;
    uint32_t meshId = -1;
    if (mDelMesh.empty())
    {
        meshId = mMeshes.size(); // add mesh to storage
        mMeshes.push_back({});
        mesh = &mMeshes.back();
    }
    else
    {
        meshId = mDelMesh.top(); // get index from stack
        mDelMesh.pop(); // del taken index from stack
        mesh = &mMeshes[meshId];
    }

    mesh->mIndex = mIndices.size(); // Index of 1st index in index buffer
    mesh->mCount = ib.size(); // amount of indices in mesh

    mesh->mVbOffset = mVertices.size();
    mesh->mVertexCount = vb.size();

    // const uint32_t ibOffset = mVertices.size(); // adjust indices for global index buffer
    // for (int i = 0; i < ib.size(); ++i)
    // {
    //     mIndices.push_back(ibOffset + ib[i]);
    // }
    mIndices.insert(mIndices.end(), ib.begin(), ib.end());
    mVertices.insert(mVertices.end(), vb.begin(), vb.end()); // copy vertices
    return meshId;
}

uint32_t Scene::createMesh(const std::vector<Vertex>& vb, const std::vector<uint32_t>& ib, const std::vector<oka::Scene::vertexSkinData>& sb)
{
    std::scoped_lock lock(mMeshMutex);

    Mesh* mesh = nullptr;
    uint32_t meshId = -1;
    if (mDelMesh.empty())
    {
        meshId = mMeshes.size(); // add mesh to storage
        mMeshes.push_back({});
        mesh = &mMeshes.back();
    }
    else
    {
        meshId = mDelMesh.top(); // get index from stack
        mDelMesh.pop(); // del taken index from stack
        mesh = &mMeshes[meshId];
    }

    mesh->mIndex = mIndices.size(); // Index of 1st index in index buffer
    mesh->mCount = ib.size(); // amount of indices in mesh

    mesh->mVbOffset = mVertices.size();
    mesh->mVertexCount = vb.size();

    mesh->mSbOffset = mVertexSkinData.size();

    // const uint32_t ibOffset = mVertices.size(); // adjust indices for global index buffer
    // for (int i = 0; i < ib.size(); ++i)
    // {
    //     mIndices.push_back(ibOffset + ib[i]);
    // }
    mIndices.insert(mIndices.end(), ib.begin(), ib.end());
    mVertices.insert(mVertices.end(), vb.begin(), vb.end()); // copy vertices
    mVertexSkinData.insert(mVertexSkinData.end(), sb.begin(), sb.end());
    return meshId;
}

uint32_t Scene::createInstance(const Instance::Type type,
                               const uint32_t geomId,
                               const uint32_t materialId,
                               const glm::mat4& transform,
                               const uint32_t lightId)
{
    std::scoped_lock lock(mInstanceMutex);

    Instance* inst = nullptr;
    uint32_t instId = -1;
    if (mDelInstances.empty())
    {
        instId = mInstances.size(); // add instance to storage
        mInstances.push_back({});
        inst = &mInstances.back();
    }
    else
    {
        instId = mDelInstances.top(); // get index from stack
        mDelInstances.pop(); // del taken index from stack
        inst = &mInstances[instId];
    }
    inst->type = type;
    if (inst->type == Instance::Type::eMesh || inst->type == Instance::Type::eLight)
    {
        inst->mMeshId = geomId;
    }
    else if (inst->type == Instance::Type::eCurve)
    {
        inst->mCurveId = geomId;
    }
    inst->mMaterialId = materialId;
    inst->transform = transform;
    inst->mLightId = lightId;

    mOpaqueInstances.push_back(instId);

    return instId;
}

uint32_t Scene::addMaterial(const MaterialDescription& material)
{
    // TODO: fix here
    uint32_t res = mMaterialsDescs.size();
    mMaterialsDescs.push_back(material);
    return res;
}

std::string Scene::getSceneFileName()
{
    fs::path p(modelPath);
    return p.filename().string();
};

std::string Scene::getSceneDir()
{
    fs::path p(modelPath);
    return p.parent_path().string();
}

glm::quat Scene::makeQuatFromFloat4(const glm::float4 &value)
{
    const float floatRotation[4] = {
                value[3],
                value[0],
                value[1],
                value[2],
            };
    return glm::make_quat(floatRotation);
}

glm::float4 Scene::makeFloat4FromQuat(const glm::quat &q)
{
    return glm::float4(q.x, q.y, q.z, q.w);
}

glm::float4 Scene::interpolate(const AnimationSampler &sampler, const AnimationChannel::PathType targetProperty, const float time)
{
    glm::float4 result;
    float previousTime = -std::numeric_limits<float>::max();
    float nextTime = std::numeric_limits<float>::max();
    glm::float4 previousValue, nextValue;

    for (int i = 0; i < sampler.inputs.size(); ++i)
    {
        if (sampler.inputs[i] == time) return sampler.outputsVec4[i]; // dont need to interpolate

        if (sampler.inputs[i] < time && sampler.inputs[i] > previousTime) 
        {
            previousTime = sampler.inputs[i];
            previousValue = sampler.outputsVec4[i];
        }
        if (sampler.inputs[i] > time && sampler.inputs[i] < nextTime)
        {
            nextTime = sampler.inputs[i];
            nextValue = sampler.outputsVec4[i];
        }
    }

    switch (sampler.interpolation)
    {
    case AnimationSampler::InterpolationType::STEP :
        result = previousValue;
        break;

    case AnimationSampler::InterpolationType::CUBICSPLINE :
        std::cout << "CUBICSPLINE interpolation not yet supported, skipping" << std::endl;
        break;
    
    default: //linear
        float interpolationValue = (time - previousTime) / (nextTime - previousTime);
        if (targetProperty != AnimationChannel::PathType::ROTATION) result = glm::lerp(previousValue, nextValue, interpolationValue);
        else result = makeFloat4FromQuat(glm::slerp(makeQuatFromFloat4(previousValue), makeQuatFromFloat4(nextValue), interpolationValue));
        break;
    }
    return result;
}

bool Scene::applyAnimation(const uint32_t animId)
{
    bool blasChanged = false;
    auto &animation = mAnimations[animId];
    for (int i = 0; i < animation.channels.size(); ++i)
    {
        const uint32_t nodeId = animation.channels[i].node;
        const AnimationChannel::PathType targetProperty = animation.channels[i].path;
        const glm::float4 value = interpolate(animation.samplers[animation.channels[i].samplerIndex], targetProperty, animation.current);

        if (targetProperty == AnimationChannel::PathType::ROTATION) 
            blasChanged = animateNode(nodeId, targetProperty, makeQuatFromFloat4(value)) ? true : blasChanged;
        else 
            blasChanged = animateNode(nodeId, targetProperty, glm::float3(value)) ? true : blasChanged;
    }
    return blasChanged;
}

uint32_t packNormal(const glm::float3& normal);

void Scene::applySkinning()
{
    for (int i = 0; i < mNodes.size(); ++i)
    {
        if (mNodes[i].skin != -1 && mNodes[i].type == Node::NodeType::mesh)
        {
            auto jointCount = mSkines[mNodes[i].skin].joints.size();
            std::vector<glm::mat4> jointMat;
            computeJointMatrices(&jointMat, jointCount, mNodes[i].skin);
            for (const auto instId: mNodes[i].instanceIds) {
                auto &mesh = mMeshes[mInstances[instId].mMeshId];
                int vbOffset = mesh.mVbOffset;
                int sbOffset = mesh.mSbOffset;
                for (int iv = 0; iv < mesh.mVertexCount; ++iv)
                {
                    glm::vec4 v_weight = mVertexSkinData[sbOffset + iv].weights;
                    glm::u16vec4 v_joint = mVertexSkinData[sbOffset + iv].joints;
                    glm::mat4 skinMat = v_weight[0] * jointMat[v_joint[0]]
                                      + v_weight[1] * jointMat[v_joint[1]]
                                      + v_weight[2] * jointMat[v_joint[2]]
                                      + v_weight[3] * jointMat[v_joint[3]];
                    mVertices[vbOffset + iv].pos = skinMat * glm::vec4(mVertexSkinData[sbOffset + iv].pos, 1.0);
                    mVertices[vbOffset + iv].normal = packNormal(glm::normalize(glm::vec3(glm::mat3(skinMat) * glm::vec4(mVertexSkinData[sbOffset + iv].normal, 1.0))));
                }
            }
        }
    }
}

void Scene::computeJointMatrices(std::vector<glm::mat4> *jointMatrices, int jointCount, const uint32_t skinId)
{
    auto &skin = mSkines[skinId];
    for (int i = 0; i < jointCount; ++i)
    {
        glm::mat4 jointGlobalTransform = calculateNodeGlobalTransform(skin.joints[i]);
        jointMatrices->push_back(jointGlobalTransform * skin.inverseBindMatrices[i]);
    }
}

glm::mat4 Scene::calculateNodeLocalTransform(const uint32_t nodeId)
{
    const glm::float4x4 translationMatrix = glm::translate(glm::float4x4(1.0f), mNodes[nodeId].translation);
    const glm::float4x4 rotationMatrix{ mNodes[nodeId].rotation };
    const glm::float4x4 scaleMatrix = glm::scale(glm::float4x4(1.0f), mNodes[nodeId].scale);
    return translationMatrix * rotationMatrix * scaleMatrix;
}

glm::mat4 Scene::calculateNodeGlobalTransform(const uint32_t nodeId)
{
    int parentId = mNodes[nodeId].parent;
    if (parentId == -1) {
        return calculateNodeLocalTransform(nodeId);
    }
    else {
        return calculateNodeGlobalTransform(parentId) * calculateNodeLocalTransform(nodeId);
    }
}

bool Scene::animateNode(const uint32_t nodeId, AnimationChannel::PathType targetProperty, const glm::float3 newValue)
{
    switch (targetProperty)
    {
    case AnimationChannel::PathType::TRANSLATION:
        mNodes[nodeId].translation = newValue;
        return updateNode(nodeId);
        break;

    case AnimationChannel::PathType::SCALE:
        mNodes[nodeId].scale = newValue;
        return updateNode(nodeId);
        break;
    
    case AnimationChannel::PathType::ROTATION:
        std::cout << "Invalid value to animate ROTATION, use 2nd definition" << std::endl;
        break;

    default:
        break;
    }
    return false;
}

bool Scene::animateNode(const uint32_t nodeId, AnimationChannel::PathType targetProperty, const glm::quat newValue) 
{
    switch (targetProperty)
    {
    case AnimationChannel::PathType::TRANSLATION:
        std::cout << "Invalid value to animate TRANSLATION, use 1st definition" << std::endl;

    case AnimationChannel::PathType::SCALE:
        std::cout << "Invalid value to animate SCALE, use 1st definition" << std::endl;
        break;
    
    case AnimationChannel::PathType::ROTATION:
        mNodes[nodeId].rotation = newValue;
        return updateNode(nodeId);
        break;

    default:
        break;
    }
    return false;
}

bool Scene::updateNode(const uint32_t nodeId)
{
    bool skeletonNodesUpdated = false;
    const glm::float4x4 globalTransform = calculateNodeGlobalTransform(nodeId);

    // if this node is mesh node - updating Instances
    // if this node is skeleton node - need to rebuild blas
    switch (mNodes[nodeId].type)
    {
        case Node::NodeType::mesh:
            for (const auto instId: mNodes[nodeId].instanceIds) {
                mInstances[instId].transform = globalTransform;
            }
            return false;
            break;

        case Node::NodeType::skeleton:
            skeletonNodesUpdated = true;
            break;
        
        default:
            break;
    }

    for (const auto childId: mNodes[nodeId].children) 
    {
        skeletonNodesUpdated = updateNode(childId) ? true : skeletonNodesUpdated;
    }

    return skeletonNodesUpdated;
}

//  valid range of coordinates [-1; 1]
uint32_t packNormals(const glm::float3& normal)
{
    auto packed = (uint32_t)((normal.x + 1.0f) / 2.0f * 511.99999f);
    packed += (uint32_t)((normal.y + 1.0f) / 2.0f * 511.99999f) << 10;
    packed += (uint32_t)((normal.z + 1.0f) / 2.0f * 511.99999f) << 20;
    return packed;
}

uint32_t Scene::createRectLightMesh()
{
    if (mRectLightMeshId != -1)
    {
        return mRectLightMeshId;
    }

    std::vector<Scene::Vertex> vb;
    Scene::Vertex v1, v2, v3, v4;
    v1.pos = glm::float4(0.5f, 0.5f, 0.0f, 1.0f); // top right 0
    v2.pos = glm::float4(-0.5f, 0.5f, 0.0f, 1.0f); // top left 1
    v3.pos = glm::float4(-0.5f, -0.5f, 0.0f, 1.0f); // bottom left 2
    v4.pos = glm::float4(0.5f, -0.5f, 0.0f, 1.0f); // bottom right 3
    glm::float3 normal = glm::float3(0.f, 0.f, 1.f);
    v1.normal = v2.normal = v3.normal = v4.normal = packNormals(normal);
    std::vector<uint32_t> ib = { 0, 1, 2, 2, 3, 0 };
    vb.push_back(v1);
    vb.push_back(v2);
    vb.push_back(v3);
    vb.push_back(v4);

    uint32_t meshId = createMesh(vb, ib);
    assert(meshId != -1);

    return meshId;
}

uint32_t Scene::createSphereLightMesh()
{
    if (mSphereLightMeshId != -1)
    {
        return mSphereLightMeshId;
    }

    std::vector<Scene::Vertex> vertices;
    std::vector<uint32_t> indices;
    const int segments = 16;
    const int rings = 16;
    const float radius = 1.0f;
    // Generate vertices and normals
    for (int i = 0; i <= rings; ++i)
    {
        float theta = static_cast<float>(i) * static_cast<float>(M_PI) / static_cast<float>(rings);
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);

        for (int j = 0; j <= segments; ++j)
        {
            float phi = static_cast<float>(j) * 2.0f * static_cast<float>(M_PI) / static_cast<float>(segments);
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);

            float x = cosPhi * sinTheta;
            float y = cosTheta;
            float z = sinPhi * sinTheta;

            glm::float3 pos = { radius * x, radius * y, radius * z };
            glm::float3 normal = { x, y, z };

            vertices.push_back(Scene::Vertex{ pos, 0, packNormals(normal) });
        }
    }
    // Generate indices
    for (int i = 0; i < rings; ++i)
    {
        for (int j = 0; j < segments; ++j)
        {
            int p0 = i * (segments + 1) + j;
            int p1 = p0 + 1;
            int p2 = (i + 1) * (segments + 1) + j;
            int p3 = p2 + 1;

            indices.push_back(p0);
            indices.push_back(p1);
            indices.push_back(p2);

            indices.push_back(p2);
            indices.push_back(p1);
            indices.push_back(p3);
        }
    }
    const uint32_t meshId = createMesh(vertices, indices);
    assert(meshId != -1);

    return meshId;
}

uint32_t Scene::createDiscLightMesh()
{
    if (mDiskLightMeshId != -1)
    {
        return mDiskLightMeshId;
    }

    std::vector<Scene::Vertex> vertices;
    std::vector<uint32_t> indices;

    Scene::Vertex v1, v2;
    v1.pos = glm::float4(0.f, 0.f, 0.f, 1.f);
    v2.pos = glm::float4(1.0f, 0.f, 0.f, 1.f);

    glm::float3 normal = glm::float3(0.f, 0.f, 1.f);
    v1.normal = v2.normal = packNormals(normal);

    vertices.push_back(v1); // central point
    vertices.push_back(v2); // first point

    const float diskRadius = 1.0f; // param
    const float step = 2.0f * M_PI / 16;
    float angle = 0;
    for (int i = 0; i < 16; ++i)
    {
        indices.push_back(0); // each triangle have central point
        indices.push_back(vertices.size() - 1); // prev vertex

        angle += step;
        const float x = cos(angle) * diskRadius;
        const float y = sin(angle) * diskRadius;

        Scene::Vertex v;
        v.pos = glm::float4(x, y, 0.0f, 1.0f);
        v.normal = packNormals(normal);
        vertices.push_back(v);

        indices.push_back(vertices.size() - 1); // added vertex
    }

    uint32_t meshId = createMesh(vertices, indices);
    assert(meshId != -1);

    return meshId;
}

void Scene::updateAnimation(const float time)
{
    if (mAnimations.empty())
    {
        return;
    }
    auto& animation = mAnimations[0];
    for (auto& channel : animation.channels)
    {
        assert(channel.node < mNodes.size());
        auto& sampler = animation.samplers[channel.samplerIndex];
        if (sampler.inputs.size() > sampler.outputsVec4.size())
        {
            continue;
        }
        for (size_t i = 0; i < sampler.inputs.size() - 1; i++)
        {
            if ((time >= sampler.inputs[i]) && (time <= sampler.inputs[i + 1]))
            {
                float u = std::max(0.0f, time - sampler.inputs[i]) / (sampler.inputs[i + 1] - sampler.inputs[i]);
                if (u <= 1.0f)
                {
                    switch (channel.path)
                    {
                    case AnimationChannel::PathType::TRANSLATION: {
                        glm::vec4 trans = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], u);
                        mNodes[channel.node].translation = glm::float3(trans);
                        break;
                    }
                    case AnimationChannel::PathType::SCALE: {
                        glm::vec4 scale = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], u);
                        mNodes[channel.node].scale = glm::float3(scale);
                        break;
                    }
                    case AnimationChannel::PathType::ROTATION: {
                        float floatRotation[4] = { (float)sampler.outputsVec4[i][3], (float)sampler.outputsVec4[i][0],
                                                   (float)sampler.outputsVec4[i][1], (float)sampler.outputsVec4[i][2] };
                        float floatRotation1[4] = { (float)sampler.outputsVec4[i + 1][3],
                                                    (float)sampler.outputsVec4[i + 1][0],
                                                    (float)sampler.outputsVec4[i + 1][1],
                                                    (float)sampler.outputsVec4[i + 1][2] };
                        glm::quat q1 = glm::make_quat(floatRotation);
                        glm::quat q2 = glm::make_quat(floatRotation1);
                        mNodes[channel.node].rotation = glm::normalize(glm::slerp(q1, q2, u));
                        break;
                    }
                    }
                }
            }
        }
    }
    mCameras[0].matrices.view = getTransform(mCameras[0].node);
}

void Scene::createLight(const UniformLightDesc& desc)
{
    auto lightId = (uint32_t)mLights.size();
    Light l;
    mLights.push_back(l);
    mLightDesc.push_back(desc);

    updateLight(lightId, desc);

    // TODO: only for rect light
    // Lazy init light mesh
    glm::float4x4 scaleMatrix = glm::float4x4(0.f);
    uint32_t currentLightMeshId = 0;
    if (desc.type == 0)
    {
        mRectLightMeshId = createRectLightMesh();
        currentLightMeshId = mRectLightMeshId;
        scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(desc.width, desc.height, 1.0f));
    }
    else if (desc.type == 1)
    {
        mDiskLightMeshId = createDiscLightMesh();
        currentLightMeshId = mDiskLightMeshId;
        scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(desc.radius, desc.radius, desc.radius));
    }
    else if (desc.type == 2)
    {
        mSphereLightMeshId = createSphereLightMesh();
        currentLightMeshId = mSphereLightMeshId;
        scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(desc.radius, desc.radius, desc.radius));
    }
    else if (desc.type == 3)
    {
        // distant light
        currentLightMeshId = 0; // empty
        scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(desc.radius, desc.radius, desc.radius));
        return;
    }

    const glm::float4x4 transform = desc.useXform ? desc.xform * scaleMatrix : getTransform(desc);
    /*const glm::float4x4 transform = glm::mat4(
                                    1.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 1.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 1.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 1.0f
                                    );*/
    uint32_t instId = createInstance(Instance::Type::eLight, currentLightMeshId, (uint32_t)-1, transform, lightId);
    assert(instId != -1);

    mLightIdToInstanceId[lightId] = instId;

    //return lightId;
}

void Scene::updateLight(const uint32_t lightId, const UniformLightDesc& desc)
{
    const float intensityPerPoint = desc.intensity; // light intensity
    // transform to GPU light
    // Rect Light
    if (desc.type == 0)
    {
        const glm::float4x4 scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(desc.width, desc.height, 1.0f));
        const glm::float4x4 localTransform = desc.useXform ? desc.xform * scaleMatrix : getTransform(desc);

        mLights[lightId].points[0] = localTransform * glm::float4(0.5f, 0.5f, 0.0f, 1.0f);
        mLights[lightId].points[1] = localTransform * glm::float4(-0.5f, 0.5f, 0.0f, 1.0f);
        mLights[lightId].points[2] = localTransform * glm::float4(-0.5f, -0.5f, 0.0f, 1.0f);
        mLights[lightId].points[3] = localTransform * glm::float4(0.5f, -0.5f, 0.0f, 1.0f);

        mLights[lightId].type = 0;
    }
    else if (desc.type == 1)
    {
        // Disk Light
        const glm::float4x4 scaleMatrix =
            glm::scale(glm::float4x4(1.0f), glm::float3(desc.radius, desc.radius, desc.radius));
        const glm::float4x4 localTransform = desc.useXform ? desc.xform * scaleMatrix : getTransform(desc);

        mLights[lightId].points[0] = glm::float4(desc.radius, 0.f, 0.f, 0.f); // save radius
        mLights[lightId].points[1] = localTransform * glm::float4(0.f, 0.f, 0.f, 1.f); // save O
        mLights[lightId].points[2] = localTransform * glm::float4(1.f, 0.f, 0.f, 0.f); // OXws
        mLights[lightId].points[3] = localTransform * glm::float4(0.f, 1.f, 0.f, 0.f); // OYws

        glm::float4 normal = localTransform * glm::float4(0, 0, 1.f, 0.0f);
        mLights[lightId].normal = normal;
        mLights[lightId].type = 1;
    }
    else if (desc.type == 2)
    {
        // Sphere Light
        const glm::float4x4 scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(1.0f, 1.0f, 1.0f));
        const glm::float4x4 localTransform = desc.useXform ? scaleMatrix * desc.xform : getTransform(desc);

        mLights[lightId].points[0] = glm::float4(desc.radius, 0.f, 0.f, 0.f); // save radius
        mLights[lightId].points[1] = localTransform * glm::float4(0.f, 0.f, 0.f, 1.f); // save O

        mLights[lightId].type = 2;
    }
    else if (desc.type == 3)
    {
        // distant light https://openusd.org/release/api/class_usd_lux_distant_light.html
        mLights[lightId].type = 3;
        mLights[lightId].halfAngle = desc.halfAngle;
        const glm::float4x4 scaleMatrix = glm::float4x4(1.0f);
        const glm::float4x4 localTransform = desc.useXform ? desc.xform * scaleMatrix : getTransform(desc);
        mLights[lightId].normal = glm::normalize(localTransform * glm::float4(0.0f, 0.0f, -1.0f, 0.0f)); // -Z
    }

    mLights[lightId].color = glm::float4(desc.color, 1.0f) * intensityPerPoint;
    mDirty = DirtyFlag::eLights;
}

void Scene::removeInstance(const uint32_t instId)
{
    mDelInstances.push(instId); // marked as removed
}

void Scene::removeMesh(const uint32_t meshId)
{
    mDelMesh.push(meshId); // marked as removed
}

void Scene::removeMaterial(const uint32_t materialId)
{
    mDelMaterial.push(materialId); // marked as removed
}

std::vector<uint32_t>& Scene::getOpaqueInstancesToRender(const glm::float3& camPos)
{
    return mOpaqueInstances;
}

std::vector<uint32_t>& Scene::getTransparentInstancesToRender(const glm::float3& camPos)
{
    return mTransparentInstances;
}

std::set<uint32_t> Scene::getDirtyInstances()
{
    return this->mDirtyInstances;
}

void Scene::updateInstanceTransform(uint32_t instId, glm::float4x4 newTransform)
{
    Instance& inst = mInstances[instId];
    inst.transform = newTransform;
    mDirtyInstances.insert(instId);
}

uint32_t Scene::createCurve(const Curve::Type type,
                            const std::vector<uint32_t>& vertexCounts,
                            const std::vector<glm::float3>& points,
                            const std::vector<float>& widths)
{
    Curve c = {};
    c.mPointsStart = mCurvePoints.size();
    c.mPointsCount = points.size();
    mCurvePoints.insert(mCurvePoints.end(), points.begin(), points.end());
    c.mVertexCountsStart = mCurveVertexCounts.size();
    c.mVertexCountsCount = vertexCounts.size();
    mCurveVertexCounts.insert(mCurveVertexCounts.end(), vertexCounts.begin(), vertexCounts.end());
    if (!widths.empty())
    {
        c.mWidthsCount = widths.size();
        c.mWidthsStart = mCurveWidths.size();
        mCurveWidths.insert(mCurveWidths.end(), widths.begin(), widths.end());
    }
    else
    {
        c.mWidthsCount = -1;
        c.mWidthsStart = -1;
    }
    uint32_t res = mCurves.size();
    mCurves.push_back(c);
    return res;
}


} // namespace oka
