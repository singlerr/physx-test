#ifndef _MESHES_H_
#define _MESHES_H_

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <PxPhysicsAPI.h>
namespace meshes
{

    fastgltf::Expected<fastgltf::Asset> loadAsset(std::filesystem::path path);
    physx::PxSoftBodyMesh* createSoftBodyMesh(fastgltf::Expected<fastgltf::Asset>& asset,physx::PxCookingParams& params, physx::PxPhysics* physics, std::function<bool(fastgltf::Node&)> nodeSelector);
    physx::PxRigidDynamic* createDynamic(fastgltf::Expected<fastgltf::Asset>& asset, physx::PxCookingParams& params,
                                           physx::PxPhysics* physics, physx::PxMaterial* material, physx::PxScene* scene, physx::PxCudaContextManager* cudaContextManager,
                                           std::function<bool(fastgltf::Node&)> nodeSelector);
}

#endif