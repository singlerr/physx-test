#ifndef _MESHES_H_
#define _MESHES_H_

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <PxPhysicsAPI.h>
#include <cudamanager/PxCudaContext.h>

namespace meshes
{
    using SDFParams = struct SDFParams
    {
        physx::PxReal spacing = 0.05f;
        physx::PxU32 subgridSize = 6;
        physx::PxSdfBitsPerSubgridPixel::Enum bitsPerSubgridPixel = physx::PxSdfBitsPerSubgridPixel::e16_BIT_PER_PIXEL;
        physx::PxU32 numThreadsForSdfConstruction = 8;
    };

    using DynamicParams = struct DynamicParams
    {
        physx::PxReal linearDamping = 0.2f;
        physx::PxReal angularDamping = 0.1f;
        physx::PxReal contactOffset = 0.05f;
        physx::PxReal restOffset = 0.0f;
        physx::PxReal wakeCounter = 100000000.f;
        physx::PxU32 minPositionIter = 50;
        physx::PxU32 minVelocityIter = 1;
        physx::PxReal maxDepenetrationVelocity = 5.0f;
        physx::PxReal density = 100.0f;
    };

    using ClothParams = struct ClothParams
    {
        physx::PxReal stiffness = 100.0f;
        physx::PxReal damping = 0.01f;
    };

    using ParticleParams = struct my_struct
    {
        physx::PxReal restOffset = 0.2f;
        physx::PxReal contactOffset = 0.2f;
        physx::PxReal particleContactOffset = 0.22f;
        physx::PxReal solidRestOffset = 0.2f;
        physx::PxReal fluidRestOffset = 0.0f;
    };

    class SoftBody
    {
    public:
        SoftBody(physx::PxSoftBody* softBody, physx::PxCudaContextManager* cudaContextManager) :
            mSoftBody(softBody),
            mCudaContextManager(cudaContextManager)
        {
            mPositionsInvMass = cudaContextManager->allocPinnedHostBuffer<physx::PxVec4>(
                softBody->getCollisionMesh()->getNbVertices());
        }

        ~SoftBody()
        {
        }

        void release()
        {
            if (mSoftBody)
                mSoftBody->release();

            PX_PINNED_HOST_FREE(mCudaContextManager, mPositionsInvMass);
            PX_PINNED_HOST_FREE(mCudaContextManager, mTargetPositionsH);
            PX_PINNED_HOST_FREE(mCudaContextManager, mTargetPositionsD);
        }

        void copyDeformedVerticesFromGPUAsync(CUstream stream)
        {
            physx::PxTetrahedronMesh* tetMesh = mSoftBody->getCollisionMesh();

            physx::PxScopedCudaLock _lock(*mCudaContextManager);
            mCudaContextManager->getCudaContext()->memcpyDtoHAsync(mPositionsInvMass,
                                                                   reinterpret_cast<CUdeviceptr>(mSoftBody->
                                                                       getPositionInvMassBufferD()),
                                                                   tetMesh->getNbVertices() * sizeof(physx::PxVec4),
                                                                   stream);
        }

        void copyDeformedVerticesFromGPU()
        {
            physx::PxTetrahedronMesh* tetMesh = mSoftBody->getCollisionMesh();

            physx::PxScopedCudaLock _lock(*mCudaContextManager);
            mCudaContextManager->getCudaContext()->memcpyDtoH(mPositionsInvMass,
                                                              reinterpret_cast<CUdeviceptr>(mSoftBody->
                                                                  getPositionInvMassBufferD()),
                                                              tetMesh->getNbVertices() * sizeof(physx::PxVec4));
        }


        physx::PxVec4* mPositionsInvMass;
        physx::PxSoftBody* mSoftBody;
        physx::PxCudaContextManager* mCudaContextManager;

        physx::PxVec4* mTargetPositionsH;
        physx::PxVec4* mTargetPositionsD;
        physx::PxU32 mTargetCount;
    };

    fastgltf::Expected<fastgltf::Asset> loadAsset(std::filesystem::path path);
    physx::PxSoftBodyMesh* createSoftBodyTriangleMesh(fastgltf::Expected<fastgltf::Asset>& asset,
                                                      physx::PxCookingParams& params, physx::PxPhysics* physics,
                                                      std::function<bool(fastgltf::Node&)> nodeSelector);
    physx::PxTriangleMesh* createSDFTriangleMesh(fastgltf::Expected<fastgltf::Asset>& asset,
                                                 physx::PxCookingParams& params, SDFParams sdfParams,
                                                 std::function<bool(fastgltf::Node&)> nodeSelector);
    SoftBody* createKineticSoftBody(fastgltf::Expected<fastgltf::Asset>& asset, physx::PxCookingParams& params,
                                    SDFParams sdfParams, physx::PxPhysics* physics, physx::PxScene* scene,
                                    physx::PxCudaContextManager* cudaContextManager,
                                    std::function<bool(fastgltf::Node&)> nodeSelector);
    physx::PxRigidDynamic* createTriangleDynamic(fastgltf::Expected<fastgltf::Asset>& asset,
                                                 physx::PxCookingParams& params, SDFParams sdfParams,
                                                 DynamicParams dynamicParams,
                                                 physx::PxPhysics* physics, physx::PxMaterial* material,
                                                 physx::PxScene* scene, physx::PxTransform& t,
                                                 std::function<bool(fastgltf::Node&)> nodeSelector);
    physx::PxParticleClothBuffer* createTriangleFEMCloth(fastgltf::Expected<fastgltf::Asset>& asset,
                                                         physx::PxCookingParams& params, SDFParams sdfParams,
                                                         ClothParams clothParams,
                                                         physx::PxCudaContextManager* cudaContextManager,
                                                         physx::PxPhysics* physics, physx::PxPBDMaterial* material,
                                                         physx::PxParticleSystem* particleSystem, physx::PxTransform& t,
                                                         std::function<bool(fastgltf::Node&)> nodeSelector);
    physx::PxParticleClothBuffer* createTrianglePBDCloth(fastgltf::Expected<fastgltf::Asset>& asset,
                                                         physx::PxCookingParams& params, SDFParams sdfParams,
                                                         ClothParams clothParams,
                                                         physx::PxCudaContextManager* cudaContextManager,
                                                         physx::PxPhysics* physics, physx::PxPBDMaterial* material,
                                                         physx::PxParticleSystem* particleSystem, physx::PxTransform& t,
                                                         std::function<bool(fastgltf::Node&)> nodeSelector);
}

#endif
