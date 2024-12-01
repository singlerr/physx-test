#include "meshes.h"
#include <functional>
#include <fastgltf/tools.hpp>
#include <extensions/PxTetMakerExt.h>
#include <extensions/PxSoftBodyExt.h>


using namespace physx;


void loadGltf(fastgltf::Expected<fastgltf::Asset>& asset, PxArray<PxVec3>& outVertices, PxArray<PxU32>& outIndices, std::function<bool(fastgltf::Node&)> filter)
{
    for (fastgltf::Node& node : asset->nodes)
    {
        if (! filter(node))
            continue;
        if (node.meshIndex.has_value())
        {
            size_t meshIndex = node.meshIndex.value();
            fastgltf::Mesh& mesh = asset->meshes[meshIndex];
            for (fastgltf::Primitive& primitive : mesh.primitives)
            {
                
                fastgltf::Accessor& indicesAccessor = asset->accessors[primitive.indicesAccessor.value()];
                // Load indices
                fastgltf::iterateAccessorWithIndex<uint32_t>(asset.get(), indicesAccessor, [&](uint32_t value, std::size_t idx)
                {
                    outIndices.pushBack(value);
                });

                fastgltf::Attribute* posAttr = primitive.findAttribute("POSITION");
                fastgltf::Accessor& posAccessor = asset->accessors[posAttr->accessorIndex];
                fastgltf::iterateAccessor<fastgltf::math::fvec3>(asset.get(), posAccessor, [&](fastgltf::math::fvec3 pos)
                {
                    outVertices.pushBack(PxVec3(pos[0], pos[1], pos[2]));
                });
            } 
        }
    } 
}

fastgltf::Expected<fastgltf::Asset> meshes::loadAsset(std::filesystem::path path)
{
    fastgltf::Parser parser;

    auto data = fastgltf::GltfDataBuffer::FromPath(path);
    fastgltf::Expected<fastgltf::Asset> asset = parser.loadGltf(data.get(), path);

    
    return asset;
}

PxSoftBodyMesh* meshes::createSoftBodyMesh(fastgltf::Expected<fastgltf::Asset>& asset,PxCookingParams& params,PxPhysics* physics, std::function<bool(fastgltf::Node&)> nodeSelector)
{
    PxArray<PxVec3> inputVertices;
    PxArray<PxU32> inputIndices;
    PxArray<PxVec3> simplifiedVertices;
    PxArray<PxU32> simplifiedIndices;
    loadGltf(asset, inputVertices, inputIndices, nodeSelector);

    PxI32 targetTriangleCount = 500;
    PxReal maximalTriangleEdgeLength = 0.0f;

    PxTetMaker::simplifyTriangleMesh(inputVertices, inputIndices, targetTriangleCount, maximalTriangleEdgeLength, simplifiedVertices, simplifiedIndices);


    PxSimpleTriangleMesh surfaceMesh;
    surfaceMesh.points.count = simplifiedVertices.size();
    surfaceMesh.points.data = simplifiedVertices.begin();
    surfaceMesh.triangles.count = simplifiedIndices.size() / 3;
    surfaceMesh.triangles.data = simplifiedIndices.begin();
    
    PxArray<PxVec3> collisionVertices,simulationVertices;
    PxArray<PxU32> collisionIndices, simulationIndices;

    PxTetMaker::createConformingTetrahedronMesh(surfaceMesh, collisionVertices, collisionIndices);
    PxTetrahedronMeshDesc meshDesc(collisionVertices, collisionIndices);

    PxU32 numVoxelsAlongLongestAABBAxis = 1;
    PxArray<PxI32> vertexToLet;
    vertexToLet.resize(meshDesc.points.count);
    PxTetMaker::createVoxelTetrahedronMesh(meshDesc, numVoxelsAlongLongestAABBAxis, simulationVertices, simulationIndices, vertexToLet.begin());
    PxTetrahedronMeshDesc simMeshDesc(simulationVertices, simulationIndices);
    PxSoftBodySimulationDataDesc simDesc(vertexToLet);

    PxSoftBodyMesh* mesh = PxCreateSoftBodyMesh(params, simMeshDesc, meshDesc, simDesc, physics->getPhysicsInsertionCallback());
    return mesh;
}


physx::PxRigidDynamic* meshes::createDynamic(fastgltf::Expected<fastgltf::Asset>& asset, physx::PxCookingParams& params,
                                           physx::PxPhysics* physics, physx::PxMaterial* material, physx::PxScene* scene,physx::PxCudaContextManager* cudaContextManager,
                                           std::function<bool(fastgltf::Node&)> nodeSelector)
{
    PxArray<PxVec3> vertices;
    PxArray<PxU32> indices;
    loadGltf(asset, vertices, indices, nodeSelector);
    
    PxTriangleMeshDesc meshDesc;
    meshDesc.points.count = vertices.size();
    meshDesc.points.data = vertices.begin();
    meshDesc.points.stride = sizeof(PxVec3);
    meshDesc.triangles.count = indices.size();
    meshDesc.triangles.data = indices.begin();
    meshDesc.triangles.stride = 3 * sizeof(PxU32);

    PxSDFDesc sdfDesc;
    sdfDesc.spacing = 0.05;
    sdfDesc.subgridSize = 6;
    sdfDesc.bitsPerSubgridPixel = PxSdfBitsPerSubgridPixel::e16_BIT_PER_PIXEL;
    sdfDesc.numThreadsForSdfConstruction = 8;
    meshDesc.sdfDesc = &sdfDesc;
    
    PxTriangleMesh* mesh = PxCreateTriangleMesh(params, meshDesc);
    PxTriangleMeshGeometry geom(mesh);
    PxRigidDynamic* body = physics->createRigidDynamic(PxTransform(PxVec3(0,1,0)));
    body->setLinearDamping(0.2f);
    body->setAngularDamping(0.1f);
    body->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES, true);
    body->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
    PxShape* shape = PxRigidActorExt::createExclusiveShape(*body, geom, *material);
    shape->setContactOffset(0.05f);
    shape->setRestOffset(0.0f);
    PxRigidBodyExt::updateMassAndInertia(*body, 100.f);
    scene->addActor(*body);
    body->setWakeCounter(100000000.f);
    body->setSolverIterationCounts(50, 1);
    body->setMaxDepenetrationVelocity(5.f);
    return body;
}



