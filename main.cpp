#include <iostream>
#include <PxPhysicsAPI.h>
#include <cudamanager/PxCudaContextManager.h>
#include <cudamanager/PxCudaContext.h>
#include <extensions/PxExtensionsAPI.h>
#include <PxFEMClothFlags.h>
#include <PxFEMClothMaterial.h>
#include <extensions/PxParticleExt.h>
#include <extensions/PxSimpleFactory.h>
#include "meshes.h"

#define PVD_HOST "127.0.0.1"

using namespace std;
using namespace physx;

static PxDefaultErrorCallback defaultErrorCallback;
static PxDefaultAllocator defaultAllocatorCallback;
static PxSimulationFilterShader defaultFilterShader = PxDefaultSimulationFilterShader;
static PxCudaContextManager* cudaContextManager;
static PxParticleSystem* particleSystem;
static PxFoundation* foundation;
static PxDefaultCpuDispatcher* dispatcher;
static PxPhysics* physics;
static PxPvd* pvd;
static PxTolerancesScale* tolerancesScale;
static PxCookingParams* cookingParams;
static PxMaterial* material;
static PxPBDMaterial* pbdMaterial;
static PxScene* scene;

static void initPhysics()
{
    foundation = PxCreateFoundation(PX_PHYSICS_VERSION, defaultAllocatorCallback, defaultErrorCallback);
    if (!foundation)
    {
        cerr << "Failed to initialize PhysX!" << endl;
    }
    PxCudaContextManagerDesc cudaContextManagerDesc;
    cudaContextManager = PxCreateCudaContextManager(*foundation, cudaContextManagerDesc, PxGetProfilerCallback());
    if (cudaContextManager && !cudaContextManager->contextIsValid())
    {
        PX_RELEASE(cudaContextManager)
        printf("Failed to initialize cuda context.\n");
    }
    pvd = PxCreatePvd(*foundation);
    PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
    pvd->connect(*transport, PxPvdInstrumentationFlag::eALL);

    PxTolerancesScale tolerances_scale;
    tolerancesScale = &tolerances_scale;
    physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, tolerances_scale, true, pvd);
    if (!physics)
    {
        cerr << "Failed to initialize PxPhysics!" << endl;
    }
    if (!PxInitExtensions(*physics, pvd))
    {
        cerr << "PxInitExtensions failed!" << endl;
    }
}

static void initCookingParams()
{
    PxCookingParams params(*tolerancesScale);
    params.meshWeldTolerance = 0.001f;
    params.meshPreprocessParams = PxMeshPreprocessingFlags(PxMeshPreprocessingFlag::eWELD_VERTICES);
    params.buildTriangleAdjacencies = false;
    params.buildGPUData = true;
    cookingParams = &params;
}

static void initScene()
{
    PxSceneDesc scene_desc(physics->getTolerancesScale());
    scene_desc.gravity = PxVec3(0.0f, -9.8f, 0.0f);

    if (!scene_desc.cpuDispatcher)
    {
        dispatcher = PxDefaultCpuDispatcherCreate(1);
        if (!dispatcher)
        {
            cerr << "PxDefaultCpuDispatcherCreate failed!" << endl;
        }
        scene_desc.cpuDispatcher = dispatcher;
    }
    if (!scene_desc.filterShader)
    {
        scene_desc.filterShader = defaultFilterShader;
    }

    if (!scene_desc.cudaContextManager)
    {
        scene_desc.cudaContextManager = cudaContextManager;
    }

    scene_desc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
    scene_desc.flags |= PxSceneFlag::eENABLE_PCM;
    scene_desc.flags |= PxSceneFlag::eENABLE_STABILIZATION;
    scene_desc.broadPhaseType = PxBroadPhaseType::eGPU;
    scene_desc.solverType = PxSolverType::eTGS;
    scene_desc.staticStructure = PxPruningStructureType::eDYNAMIC_AABB_TREE;
    scene = physics->createScene(scene_desc);
    if (!scene)
    {
        cerr << "createScene failed!" << endl;
    }
    scene->setVisualizationParameter(PxVisualizationParameter::eSIMULATION_MESH, 1.0f);
    scene->setVisualizationParameter(PxVisualizationParameter::eSCALE, 1.0f);
    scene->setVisualizationParameter(PxVisualizationParameter::eCOLLISION_SHAPES, 1.0f);
    PxPvdSceneClient* pvd_client = scene->getScenePvdClient();
    if (pvd_client)
    {
        pvd_client->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
        pvd_client->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
        pvd_client->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
    }
}

static void initParticleSystem()
{
    particleSystem = physics->createPBDParticleSystem(*cudaContextManager);
    constexpr PxReal restOffset = 0.2f;
    particleSystem->setRestOffset(restOffset);
    particleSystem->setContactOffset(restOffset + 0.02f);
    particleSystem->setParticleContactOffset(restOffset + 0.02f);
    particleSystem->setSolidRestOffset(restOffset);
    scene->addActor(*particleSystem);
}

static void initMaterial()
{
    material = physics->createMaterial(0.5, 0.5, 0.5);
    pbdMaterial = physics->createPBDMaterial(0.8f, 0.05f, 1e+6f, 0.001f, 0.5f, 0.005f, 0.05f, 0.f, 0.f);
}

meshes::SoftBody* softBody;

static void loop()
{
    constexpr PxReal dt = 1.0f / 60.0f;
    while (true)
    {
        scene->simulate(dt);
        scene->fetchResults(true);
        softBody->copyDeformedVerticesFromGPU();
    }
}

static void initActors()
{
    PxRigidStatic* ground_plane = PxCreatePlane(*physics, PxPlane(0, 1, 0, 0), *material);
    scene->addActor(*ground_plane);

    fastgltf::Expected<fastgltf::Asset> asset = meshes::loadAsset("D:\\Resources\\VRM\\jingburger.vrm");
    softBody = createKineticSoftBody(asset, *cookingParams, meshes::SDFParams(), physics, scene, cudaContextManager,
                                     [](fastgltf::Node& node)
                                     {
                                         return node.name == "Body_all";
                                     });
}

static void cleanupPhysics()
{
    PX_RELEASE(scene)
    PX_RELEASE(dispatcher)
    PX_RELEASE(physics)
    PX_RELEASE(cudaContextManager)
    if (pvd)
    {
        PxPvdTransport* transport = pvd->getTransport();
        PX_RELEASE(pvd)
        PX_RELEASE(transport)
    }

    PX_RELEASE(foundation)
}

int main(int, char**)
{
    initPhysics();
    initMaterial();
    initCookingParams();
    initScene();
    initParticleSystem();

    //----------------
    initActors();
    //----------------

    loop();

    cleanupPhysics();
    return 0;
}
