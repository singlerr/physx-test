#include "meshes.h"
#include <functional>
#include <fastgltf/tools.hpp>
#include <PxFEMClothFlags.h>
#include <extensions/PxParticleClothCooker.h>
#include <extensions/PxTetMakerExt.h>
#include <extensions/PxParticleExt.h>
#include <extensions/PxSoftBodyExt.h>
using namespace physx;


void loadGltf(fastgltf::Expected<fastgltf::Asset>& asset, PxArray<PxVec3>& outVertices, PxArray<PxU32>& outIndices,
              std::function<bool(fastgltf::Node&)> filter)
{
    for (fastgltf::Node& node : asset->nodes)
    {
        if (!filter(node))
            continue;
        if (node.meshIndex.has_value())
        {
            size_t meshIndex = node.meshIndex.value();
            fastgltf::Mesh& mesh = asset->meshes[meshIndex];
            for (fastgltf::Primitive& primitive : mesh.primitives)
            {
                fastgltf::Accessor& indicesAccessor = asset->accessors[primitive.indicesAccessor.value()];
                // Load indices
                fastgltf::iterateAccessorWithIndex<uint32_t>(asset.get(), indicesAccessor,
                                                             [&](uint32_t value, std::size_t idx)
                                                             {
                                                                 outIndices.pushBack(value);
                                                             });

                fastgltf::Attribute* posAttr = primitive.findAttribute("POSITION");
                fastgltf::Accessor& posAccessor = asset->accessors[posAttr->accessorIndex];
                fastgltf::iterateAccessor<fastgltf::math::fvec3>(asset.get(), posAccessor,
                                                                 [&](fastgltf::math::fvec3 pos)
                                                                 {
                                                                     outVertices.pushBack(
                                                                         PxVec3(pos[0], pos[1], pos[2]));
                                                                 });
            }
        }
    }
}

meshes::SoftBody* createSoftBody(PxSoftBody* softBody, PxCudaContextManager* gCudaContextManager,
                                 const PxFEMParameters& femParams, const PxFEMMaterial& /*femMaterial*/,
                                 const PxTransform& transform, const PxReal density, const PxReal scale,
                                 const PxU32 iterCount/*, PxMaterial* tetMeshMaterial*/)
{
    PxVec4* simPositionInvMassPinned;
    PxVec4* simVelocityPinned;
    PxVec4* collPositionInvMassPinned;
    PxVec4* restPositionPinned;

    PxSoftBodyExt::allocateAndInitializeHostMirror(*softBody, gCudaContextManager, simPositionInvMassPinned,
                                                   simVelocityPinned, collPositionInvMassPinned, restPositionPinned);

    constexpr PxReal maxInvMassRatio = 50.f;

    softBody->setParameter(femParams);
    //softBody->setMaterial(femMaterial);
    softBody->setSolverIterationCounts(iterCount);

    PxSoftBodyExt::transform(*softBody, transform, scale, simPositionInvMassPinned, simVelocityPinned,
                             collPositionInvMassPinned, restPositionPinned);
    PxSoftBodyExt::updateMass(*softBody, density, maxInvMassRatio, simPositionInvMassPinned);
    PxSoftBodyExt::copyToDevice(*softBody, PxSoftBodyDataFlag::eALL, simPositionInvMassPinned, simVelocityPinned,
                                collPositionInvMassPinned, restPositionPinned);

    auto sBody = new meshes::SoftBody(softBody, gCudaContextManager);

    PX_PINNED_HOST_FREE(gCudaContextManager, simPositionInvMassPinned);
    PX_PINNED_HOST_FREE(gCudaContextManager, simVelocityPinned);
    PX_PINNED_HOST_FREE(gCudaContextManager, collPositionInvMassPinned);
    PX_PINNED_HOST_FREE(gCudaContextManager, restPositionPinned);

    return sBody;
}


fastgltf::Expected<fastgltf::Asset> meshes::loadAsset(std::filesystem::path path)
{
    fastgltf::Parser parser;

    auto data = fastgltf::GltfDataBuffer::FromPath(path);
    fastgltf::Expected<fastgltf::Asset> asset = parser.loadGltf(data.get(), path);
    return asset;
}

PxSoftBodyMesh* meshes::createSoftBodyTriangleMesh(fastgltf::Expected<fastgltf::Asset>& asset, PxCookingParams& params,
                                                   PxPhysics* physics,
                                                   std::function<bool(fastgltf::Node&)> nodeSelector)
{
    PxArray<PxVec3> inputVertices;
    PxArray<PxU32> inputIndices;
    PxArray<PxVec3> simplifiedVertices;
    PxArray<PxU32> simplifiedIndices;
    loadGltf(asset, inputVertices, inputIndices, nodeSelector);

    PxI32 targetTriangleCount = 500;
    PxReal maximalTriangleEdgeLength = 0.0f;

    PxTetMaker::simplifyTriangleMesh(inputVertices, inputIndices, targetTriangleCount, maximalTriangleEdgeLength,
                                     simplifiedVertices, simplifiedIndices);


    PxSimpleTriangleMesh surfaceMesh;
    surfaceMesh.points.count = simplifiedVertices.size();
    surfaceMesh.points.data = simplifiedVertices.begin();
    surfaceMesh.triangles.count = simplifiedIndices.size() / 3;
    surfaceMesh.triangles.data = simplifiedIndices.begin();

    PxArray<PxVec3> collisionVertices, simulationVertices;
    PxArray<PxU32> collisionIndices, simulationIndices;

    PxTetMaker::createConformingTetrahedronMesh(surfaceMesh, collisionVertices, collisionIndices);
    PxTetrahedronMeshDesc meshDesc(collisionVertices, collisionIndices);

    PxU32 numVoxelsAlongLongestAABBAxis = 1;
    PxArray<PxI32> vertexToLet;
    vertexToLet.resize(meshDesc.points.count);
    PxTetMaker::createVoxelTetrahedronMesh(meshDesc, numVoxelsAlongLongestAABBAxis, simulationVertices,
                                           simulationIndices, vertexToLet.begin());
    PxTetrahedronMeshDesc simMeshDesc(simulationVertices, simulationIndices);
    PxSoftBodySimulationDataDesc simDesc(vertexToLet);

    PxSoftBodyMesh* mesh = PxCreateSoftBodyMesh(params, simMeshDesc, meshDesc, simDesc,
                                                physics->getPhysicsInsertionCallback());
    return mesh;
}

PxTriangleMesh* meshes::createSDFTriangleMesh(fastgltf::Expected<fastgltf::Asset>& asset, PxCookingParams& params,
                                              SDFParams sdfParams, std::function<bool(fastgltf::Node&)> nodeSelector)
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
    sdfDesc.spacing = sdfParams.spacing;
    sdfDesc.subgridSize = sdfParams.subgridSize;
    sdfDesc.bitsPerSubgridPixel = sdfParams.bitsPerSubgridPixel;
    sdfDesc.numThreadsForSdfConstruction = sdfParams.numThreadsForSdfConstruction;
    meshDesc.sdfDesc = &sdfDesc;
    PxTolerancesScale tolerancesScale;
    PxCookingParams cookingParams(tolerancesScale);
    cookingParams.meshWeldTolerance = 0.001f;
    cookingParams.meshPreprocessParams = PxMeshPreprocessingFlags(PxMeshPreprocessingFlag::eWELD_VERTICES);
    cookingParams.buildTriangleAdjacencies = false;
    cookingParams.buildGPUData = true;
    return PxCreateTriangleMesh(cookingParams, meshDesc);
}

meshes::SoftBody* meshes::createKineticSoftBody(fastgltf::Expected<fastgltf::Asset>& asset,
                                                PxCookingParams& p, SDFParams sdfParams, PxPhysics* physics,
                                                PxScene* scene, PxCudaContextManager* cudaContextManager,
                                                std::function<bool(fastgltf::Node&)> nodeSelector)
{
    PxTolerancesScale scale;
    PxCookingParams params(scale);
    params.meshWeldTolerance = 0.001f;
    params.meshPreprocessParams = PxMeshPreprocessingFlags(PxMeshPreprocessingFlag::eWELD_VERTICES);
    params.buildTriangleAdjacencies = false;
    params.buildGPUData = true;


    PxArray<PxVec3> inputVertices;
    PxArray<PxU32> inputIndices;
    PxArray<PxVec3> simplifiedVertices;
    PxArray<PxU32> simplifiedIndices;
    loadGltf(asset, inputVertices, inputIndices, nodeSelector);

    PxI32 targetTriangleCount = 500;
    PxReal maximalTriangleEdgeLength = 1.0f;

    PxTetMaker::simplifyTriangleMesh(inputVertices, inputIndices, targetTriangleCount, maximalTriangleEdgeLength,
                                     simplifiedVertices, simplifiedIndices);


    PxSimpleTriangleMesh surfaceMesh;
    surfaceMesh.points.count = simplifiedVertices.size();
    surfaceMesh.points.data = simplifiedVertices.begin();
    surfaceMesh.triangles.count = simplifiedIndices.size() / 3;
    surfaceMesh.triangles.data = simplifiedIndices.begin();

    PxArray<PxVec3> collisionVertices, simulationVertices;
    PxArray<PxU32> collisionIndices, simulationIndices;

    PxTetMaker::createConformingTetrahedronMesh(surfaceMesh, collisionVertices, collisionIndices);
    PxTetrahedronMeshDesc meshDesc(collisionVertices, collisionIndices);

    PxU32 numVoxelsAlongLongestAABBAxis = 1;
    PxArray<PxI32> vertexToLet;
    vertexToLet.resize(meshDesc.points.count);
    PxTetMaker::createVoxelTetrahedronMesh(meshDesc, numVoxelsAlongLongestAABBAxis, simulationVertices,
                                           simulationIndices, vertexToLet.begin());
    PxTetrahedronMeshDesc simMeshDesc(simulationVertices, simulationIndices);
    PxSoftBodySimulationDataDesc simDesc(vertexToLet);

    PxSoftBodyMesh* softBodyMesh = PxCreateSoftBodyMesh(params, simMeshDesc, meshDesc, simDesc,
                                                        *PxGetStandaloneInsertionCallback());

    PxFEMSoftBodyMaterial* material = PxGetPhysics().createFEMSoftBodyMaterial(1e+6f, 0.45f, 0.5f);
    material->setDamping(0.005f);

    PX_ASSERT(softBodyMesh);

    PxSoftBody* softBody = physics->createSoftBody(*cudaContextManager);
    PxShapeFlags shapeFlags = PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSCENE_QUERY_SHAPE |
        PxShapeFlag::eSIMULATION_SHAPE;
    PxFEMSoftBodyMaterial* materialPtr = PxGetPhysics().createFEMSoftBodyMaterial(1e+6f, 0.45f, 0.5f);
    materialPtr->setMaterialModel(PxFEMSoftBodyMaterialModel::eNEO_HOOKEAN);
    PxTetrahedronMeshGeometry geometry(softBodyMesh->getCollisionMesh());
    PxShape* shape = physics->createShape(geometry, &materialPtr, 1, true, shapeFlags);
    softBody->attachShape(*shape);
    shape->setSimulationFilterData(PxFilterData(0, 0, 2, 0));

    softBody->attachSimulationMesh(*softBodyMesh->getSimulationMesh(), *softBodyMesh->getSoftBodyAuxData());

    scene->addActor(*softBody);

    PxFEMParameters femParams;
    SoftBody* body = createSoftBody(softBody, cudaContextManager, femParams, *material,
                                    PxTransform(PxVec3(0, 0, 0), PxQuat(PxIdentity)), 100.0f, 1.0f, 30);
    return body;
}

PxRigidDynamic* meshes::createTriangleDynamic(fastgltf::Expected<fastgltf::Asset>& asset, PxCookingParams& params,
                                              SDFParams sdfParams, DynamicParams dynamicParams,
                                              PxPhysics* physics, PxMaterial* material, PxScene* scene, PxTransform& t,
                                              std::function<bool(fastgltf::Node&)> nodeSelector)
{
    PxTriangleMesh* mesh = createSDFTriangleMesh(asset, params, sdfParams, nodeSelector);
    PxTriangleMeshGeometry geom(mesh);
    PxRigidDynamic* body = physics->createRigidDynamic(t);
    body->setLinearDamping(dynamicParams.linearDamping);
    body->setAngularDamping(dynamicParams.angularDamping);
    body->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES, true);
    body->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
    PxShape* shape = PxRigidActorExt::createExclusiveShape(*body, geom, *material);
    shape->setContactOffset(dynamicParams.contactOffset);
    shape->setRestOffset(dynamicParams.restOffset);
    PxRigidBodyExt::updateMassAndInertia(*body, dynamicParams.density);
    scene->addActor(*body);
    body->setWakeCounter(dynamicParams.wakeCounter);
    body->setSolverIterationCounts(dynamicParams.minPositionIter, dynamicParams.minVelocityIter);
    body->setMaxDepenetrationVelocity(dynamicParams.maxDepenetrationVelocity);
    return body;
}

PxParticleClothBuffer* meshes::createTriangleFEMCloth(fastgltf::Expected<fastgltf::Asset>& asset,
                                                      PxCookingParams& params, SDFParams sdfParams,
                                                      ClothParams clothParams,
                                                      PxCudaContextManager* cudaContextManager, PxPhysics* physics,
                                                      PxPBDMaterial* material,
                                                      PxParticleSystem* particleSystem, PxTransform& t,
                                                      std::function<bool(fastgltf::Node&)> nodeSelector)
{
    PxArray<PxVec3> vertices;
    PxArray<PxU32> indices;
    loadGltf(asset, vertices, indices, nodeSelector);
    PxFEMCloth* cloth = physics->createFEMCloth(*cudaContextManager);
    return nullptr;
}

PxParticleClothBuffer* meshes::createTrianglePBDCloth(fastgltf::Expected<fastgltf::Asset>& asset,
                                                      PxCookingParams& params, SDFParams sdfParams,
                                                      ClothParams clothParams, PxCudaContextManager* cudaContextManager,
                                                      PxPhysics* physics, PxPBDMaterial* material,
                                                      PxParticleSystem* particleSystem, PxTransform& t,
                                                      std::function<bool(fastgltf::Node&)> nodeSelector)
{
    PxArray<PxVec3> vertices;
    PxArray<PxU32> indices;
    loadGltf(asset, vertices, indices, nodeSelector);

    PxArray<PxVec4> quadVertices;
    quadVertices.resize(vertices.size());
    for (PxVec3 vertex : vertices)
    {
        quadVertices.pushBack(PxVec4(vertex.x, vertex.y, vertex.z, 1.0f));
    }

    ExtGpu::PxParticleClothCooker* cooker = PxCreateParticleClothCooker(quadVertices.size(), quadVertices.begin(),
                                                                        indices.size(), indices.begin(),
                                                                        ExtGpu::PxParticleClothConstraint::eTYPE_HORIZONTAL_CONSTRAINT
                                                                        | ExtGpu::PxParticleClothConstraint::eTYPE_VERTICAL_CONSTRAINT
                                                                        | ExtGpu::PxParticleClothConstraint::eTYPE_DIAGONAL_CONSTRAINT);
    cooker->cookConstraints();
    cooker->calculateMeshVolume();
    PxArray<PxU32> triangles;
    PxArray<PxParticleSpring> springs;

    PxU32 cookedTriangleIndicesCount = cooker->getTriangleIndicesCount();
    PxU32* cookedTriangleIndices = cooker->getTriangleIndices();
    for (PxU32 t = 0; t < cookedTriangleIndicesCount; t += 3)
    {
        triangles.pushBack(cookedTriangleIndices[t + 0]);
        triangles.pushBack(cookedTriangleIndices[t + 1]);
        triangles.pushBack(cookedTriangleIndices[t + 2]);
    }

    PxU32 constraintCount = cooker->getConstraintCount();
    ExtGpu::PxParticleClothConstraint* constraintBuffer = cooker->getConstraints();
    for (PxU32 i = 0; i < constraintCount; i++)
    {
        ExtGpu::PxParticleClothConstraint constraint = constraintBuffer[i];
        PxParticleSpring spring;
        spring.ind0 = constraint.particleIndexA;
        spring.ind1 = constraint.particleIndexB;
        spring.stiffness = clothParams.stiffness;
        spring.length = constraint.length;
        spring.damping = clothParams.damping;
        springs.pushBack(spring);
    }

    PxU32 numParticles = cookedTriangleIndicesCount / 3;
    PxU32* phase = cudaContextManager->allocPinnedHostBuffer<PxU32>(numParticles, PX_FL);
    PxVec4* positionInvMass = cudaContextManager->allocPinnedHostBuffer<PxVec4>(numParticles, PX_FL);
    PxVec4* velocity = cudaContextManager->allocPinnedHostBuffer<PxVec4>(numParticles, PX_FL);

    const PxU32 particlePhase = particleSystem->createPhase(
        material, PxParticlePhaseFlags(
            PxParticlePhaseFlag::eParticlePhaseSelfCollideFilter | PxParticlePhaseFlag::eParticlePhaseSelfCollide));
    for (int i = 0; i < numParticles; i++)
    {
        phase[i] = particlePhase;
        auto pos = PxVec3(cookedTriangleIndices[i], cookedTriangleIndices[i + 1], cookedTriangleIndices[i + 2]);
        positionInvMass[i] = PxVec4(pos.x, pos.y, pos.z, 0.01);
        velocity[i] = PxVec4(0.0f);
    }

    ExtGpu::PxParticleClothBufferHelper* clothBuffers = ExtGpu::PxCreateParticleClothBufferHelper(
        5, triangles.size(), springs.size(), numParticles, cudaContextManager);
    clothBuffers->addCloth(0.0f, 0.0f, 0.0f, triangles.begin(), triangles.size() / 3, springs.begin(), springs.size(),
                           positionInvMass, numParticles);

    ExtGpu::PxParticleBufferDesc bufferDesc;
    bufferDesc.maxParticles = numParticles;
    bufferDesc.numActiveParticles = numParticles;
    bufferDesc.positions = positionInvMass;
    bufferDesc.velocities = velocity;
    bufferDesc.phases = phase;

    const PxParticleClothDesc& clothDesc = clothBuffers->getParticleClothDesc();
    PxParticleClothPreProcessor* clothPreProcessor = PxCreateParticleClothPreProcessor(cudaContextManager);

    PxPartitionedParticleCloth output;
    clothPreProcessor->partitionSprings(clothDesc, output);
    clothPreProcessor->release();

    PxParticleClothBuffer* clothBuffer = PxCreateAndPopulateParticleClothBuffer(
        bufferDesc, clothDesc, output, cudaContextManager);
    PX_PINNED_HOST_FREE(cudaContextManager, positionInvMass)
    PX_PINNED_HOST_FREE(cudaContextManager, velocity)
    PX_PINNED_HOST_FREE(cudaContextManager, phase)
    return clothBuffer;
}
