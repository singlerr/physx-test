#include <iostream>
#include <PxPhysicsAPI.h>
#include <cudamanager/PxCudaContextManager.h>
#include <cudamanager/PxCudaContext.h>
#include <extensions/PxExtensionsAPI.h>
#include <extensions/PxSoftBodyExt.h>
#include <extensions\PxSimpleFactory.h>
#include "meshes.h"

#define PVD_HOST "127.0.0.1"

using namespace std;
using namespace physx;

static PxDefaultErrorCallback defaultErrorCallback;
static PxDefaultAllocator defaultAllocatorCallback;
static PxSimulationFilterShader defaultFilterShader = PxDefaultSimulationFilterShader;
static PxCudaContextManager* cudaContextManager;
static PxFoundation* foundation;
static PxPhysics* physics;
static PxPvd* pvd;

static PxScene* scene;



int main(int, char**){
    foundation = PxCreateFoundation(PX_PHYSICS_VERSION, defaultAllocatorCallback, defaultErrorCallback);
	if(! foundation) {
		cout << "Failed to initialize PhysX!" << endl;
		return 0;
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

	physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, tolerances_scale, true, pvd);
	if(! physics) {
		cerr << "Failed to initialize PxPhysics!" << endl;
		return 0;
	}
	if (! PxInitExtensions(*physics, pvd))
	{
		cerr << "PxInitExtensions failed!" << endl;
	}

	PxCookingParams params(tolerances_scale);
	params.meshWeldTolerance = 0.001f;
	params.meshPreprocessParams = PxMeshPreprocessingFlags(PxMeshPreprocessingFlag::eWELD_VERTICES);
	params.buildTriangleAdjacencies = false;
	params.buildGPUData = true;
	
	PxSceneDesc scene_desc(physics->getTolerancesScale());
	scene_desc.gravity = PxVec3(0.0f, -9.8f,0.0f);
	
	if (!scene_desc.cpuDispatcher)
	{
		PxDefaultCpuDispatcher* cpu_dispatcher = PxDefaultCpuDispatcherCreate(1);
		if (! cpu_dispatcher)
		{
			cerr << "PxDefaultCpuDispatcherCreate failed!" << endl; 
		}

		scene_desc.cpuDispatcher = cpu_dispatcher;
	}
	if (! scene_desc.filterShader)
	{
		scene_desc.filterShader = defaultFilterShader;
	}

	if (! scene_desc.cudaContextManager)
	{
		scene_desc.cudaContextManager = cudaContextManager;
	}	

	scene_desc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
	scene_desc.flags |= PxSceneFlag::eENABLE_PCM;
	scene_desc.flags |= PxSceneFlag::eENABLE_STABILIZATION;
	scene_desc.broadPhaseType = PxBroadPhaseType::eGPU;
	scene_desc.solverType = PxSolverType::eTGS;

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
	
	//-----------------------------------------------------------------------------//

	PxMaterial* material = physics->createMaterial(0.5, 0.5, 0.5);
	
	PxRigidStatic* ground_plane = PxCreatePlane(*physics, PxPlane(0,1,0,0), *material);
	scene->addActor(*ground_plane);

	fastgltf::Expected<fastgltf::Asset> asset = meshes::loadAsset("D:\\Resources\\VRM\\jingburger.vrm");
	PxRigidDynamic* body = meshes::createDynamic(asset, params, physics, material, scene, cudaContextManager, [](fastgltf::Node& node) -> bool{
		return node.name == "Body_all";
	});

    while (true)
    {
	    scene->simulate(0.01f);
    	scene->fetchResults(true);
    }
	return 0;
}

