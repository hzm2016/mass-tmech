#include "EnvManager.h"
#include "DARTHelper.h"
#include <omp.h>


EnvManager::
EnvManager(std::string meta_file,int num_envs)
	:mNumEnvs(num_envs)
{
	dart::math::seedRand();
	omp_set_num_threads(mNumEnvs);
	for(int i = 0;i<mNumEnvs;i++){
		mEnvs.push_back(new MASS::Environment());
		MASS::Environment* env = mEnvs.back();

		env->Initialize(meta_file,false);
	}
	muscle_torque_cols = mEnvs[0]->GetMuscleTorques().rows();
	tau_des_cols = mEnvs[0]->GetDesiredTorques().rows();
	mEoe.resize(mNumEnvs);
	mRewards.resize(mNumEnvs);
	mStates.resize(mNumEnvs, GetNumState());
	mMuscleTorques.resize(mNumEnvs, muscle_torque_cols);
	mDesiredTorques.resize(mNumEnvs, tau_des_cols);  

	mExoStates.resize(mNumEnvs, GetNumExoState());   
	mHumanStates.resize(mNumEnvs, GetNumHumanState());    
}

int
EnvManager::
GetNumState()
{
	return mEnvs[0]->GetNumState();
}

int
EnvManager::
GetNumExoState()
{
	return mEnvs[0]->GetNumExoState();  
}   

int
EnvManager::
GetNumHumanState()
{
	return mEnvs[0]->GetNumHumanState();    
}   

int
EnvManager::
GetNumHumanAction()
{
	return mEnvs[0]->GetNumHumanAction();  
}

int
EnvManager::
GetNumExoAction()  
{
	return mEnvs[0]->GetNumExoAction();    
}  

int
EnvManager::
GetSimulationHz()
{
	return mEnvs[0]->GetSimulationHz();
}

int
EnvManager::
GetControlHz()
{
	return mEnvs[0]->GetControlHz();
}

int
EnvManager::
GetNumSteps()
{
	return mEnvs[0]->GetNumSteps();
}

bool
EnvManager::
UseMuscle()
{
	return mEnvs[0]->GetUseMuscle();
}

void
EnvManager::
Step(int id)
{
	mEnvs[id]->Step();
}

void
EnvManager::
Reset(bool RSI,int id)
{
	mEnvs[id]->Reset(RSI);
}

bool
EnvManager::
IsEndOfEpisode(int id)
{
	return mEnvs[id]->IsEndOfEpisode();
}

double 
EnvManager::
GetReward(int id)
{
	return mEnvs[id]->GetReward();
}

double 
EnvManager::
GetExoReward(int id)
{
	return mEnvs[id]->GetExoReward();  
}

double 
EnvManager::
GetHumanReward(int id)
{
	return mEnvs[id]->GetHumanReward();  
}

const Eigen::VectorXd&
EnvManager::
GetExoActions(int id)
{
	return mEnvs[id]->GetExoAction();
}

const Eigen::VectorXd&
EnvManager::
GetHumanActions(int id)  
{
	return mEnvs[id]->GetHumanAction();
}

void
EnvManager::
Steps(int num, int doneStep)
{
	int totalStep = this->GetNumSteps();  
#pragma omp parallel for
	for (int id = 0;id<mNumEnvs;++id)
	{
		for(int j=0;j<num;j++)
			mEnvs[id]->ProcessAction(j+doneStep, totalStep);       
			mEnvs[id]->Step();
	}
}

void
EnvManager::
StepsAtOnce()
{
	int num = this->GetNumSteps();
#pragma omp parallel for
	for (int id = 0;id<mNumEnvs;++id)
	{
		for(int j=0;j<num;j++)
			mEnvs[id]->ProcessAction(j, num);   
			mEnvs[id]->Step();
	}
}

void
EnvManager::
Resets(bool RSI)
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->Reset(RSI);
	}
}

const Eigen::VectorXd&
EnvManager::
IsEndOfEpisodes()
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEoe[id] = (double)mEnvs[id]->IsEndOfEpisode();
	}

	return mEoe;
}

const Eigen::MatrixXd&
EnvManager:: 
GetStates()  
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mStates.row(id) = mEnvs[id]->GetState().transpose();
	}
	return mStates;
}

void
EnvManager::
SetHumanActions(const Eigen::MatrixXd& actions)
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->SetHumanAction(actions.row(id).transpose());  
	}
}

void
EnvManager::
SetExoHumanActions(const Eigen::MatrixXd& exoactions, const Eigen::MatrixXd& humanactions)  
{
	for (int id = 0;id<mNumEnvs;++id) 
	{
		mEnvs[id]->SetExoAction(exoactions.row(id).transpose());    
		mEnvs[id]->SetHumanAction(humanactions.row(id).transpose());       
	}
}   

const Eigen::VectorXd&
EnvManager::
GetRewards()
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mExoRewards[id] = mEnvs[id]->GetReward(); 
	}
	return mExoRewards;   
}

const Eigen::VectorXd&
EnvManager::
GetExoRewards()
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mHumanRewards[id] = mEnvs[id]->GetExoReward(); 
	}
	return mHumanRewards; 
}  

const Eigen::VectorXd&
EnvManager::
GetHumanRewards()
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mHumanRewards[id] = mEnvs[id]->GetHumanReward(); 
	}
	return mHumanRewards; 
}  

const Eigen::MatrixXd&
EnvManager:: 
GetExoStates()   
{
	for (int id = 0;id<mNumEnvs;++id)  
	{
		mExoStates.row(id) = mEnvs[id]->GetExoControlState().transpose();   
	}
	return mExoStates;   
}

const Eigen::MatrixXd&
EnvManager:: 
GetHumanStates()  
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mHumanStates.row(id) = mEnvs[id]->GetHumanState().transpose();  
	}
	return mHumanStates;     
}

const Eigen::MatrixXd&
EnvManager::
GetMuscleTorques()  
{
#pragma omp parallel for
	for (int id = 0; id < mNumEnvs; ++id)
	{
		mMuscleTorques.row(id) = mEnvs[id]->GetMuscleTorques();
	}
	return mMuscleTorques;
}

const Eigen::MatrixXd&
EnvManager::
GetDesiredTorques()
{
#pragma omp parallel for
	for (int id = 0; id < mNumEnvs; ++id)
	{
		mDesiredTorques.row(id) = mEnvs[id]->GetDesiredTorques();
	}
	return mDesiredTorques;  
}   

void
EnvManager::
SetActivationLevels(const Eigen::MatrixXd& activations)
{
	for (int id = 0; id < mNumEnvs; ++id)
		mEnvs[id]->SetActivationLevels(activations.row(id));  
}

void
EnvManager::
ComputeMuscleTuples()
{
	int n = 0;
	int rows_JtA;
	int rows_tau_des;
	int rows_L;
	int rows_b;

	for(int id=0;id<mNumEnvs;id++)
	{
		auto& tps = mEnvs[id]->GetMuscleTuples();
		n += tps.size();
		if(tps.size()!=0)
		{
			rows_JtA = tps[0].JtA.rows();
			rows_tau_des = tps[0].tau_des.rows();
			rows_L = tps[0].L.rows();
			rows_b = tps[0].b.rows();
		}
	}
	
	mMuscleTuplesJtA.resize(n, rows_JtA);
	mMuscleTuplesTauDes.resize(n, rows_tau_des);
	mMuscleTuplesL.resize(n, rows_L);
	mMuscleTuplesb.resize(n, rows_b);

	int o = 0;
	for(int id=0;id<mNumEnvs;id++)
	{
		auto& tps = mEnvs[id]->GetMuscleTuples();
		for(int j=0;j<tps.size();j++)
		{
			mMuscleTuplesJtA.row(o) = tps[j].JtA;
			mMuscleTuplesTauDes.row(o) = tps[j].tau_des;
			mMuscleTuplesL.row(o) = tps[j].L;
			mMuscleTuplesb.row(o) = tps[j].b;
			o++;
		}
		tps.clear();
	}
}

const Eigen::MatrixXd&
EnvManager::
GetMuscleTuplesJtA()
{
	return mMuscleTuplesJtA;
}
const Eigen::MatrixXd&
EnvManager::
GetMuscleTuplesTauDes()
{
	return mMuscleTuplesTauDes;
}
const Eigen::MatrixXd&
EnvManager::
GetMuscleTuplesL()
{
	return mMuscleTuplesL;
}
const Eigen::MatrixXd&
EnvManager::
GetMuscleTuplesb()
{
	return mMuscleTuplesb;
}

int
EnvManager::
GetNumFullObservation()
{
	return mEnvs[0]->GetNumFullObservation();
}

const Eigen::MatrixXd&
EnvManager::  
GetFullObservations()   
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mObservations.row(id) = mEnvs[id]->GetFullObservation().transpose();  
	}
	return mObservations;   
}

// update state and action buffers
void 
EnvManager::  
UpdateTorqueBuffer()  
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->UpdateTorqueBuffer();  
	}
}

// update state and action buffers
void 
EnvManager::  
UpdateStateBuffers()
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->UpdateStateBuffer();
	}
}

void
EnvManager::
UpdateExoActionBuffers(const Eigen::MatrixXd& exoactions)      
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->UpdateExoActionBuffer(exoactions.row(id).transpose());
	}
}

void
EnvManager::
UpdateHumanActionBuffers(const Eigen::MatrixXd& humanactions)   
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->UpdateHumanActionBuffer(humanactions.row(id).transpose());
	}
}



PYBIND11_MODULE(pymss, m)
{
	py::class_<EnvManager>(m, "pymss")
		.def(py::init<std::string,int>())
		.def("GetNumState",&EnvManager::GetNumState)
		.def("GetNumHumanState",&EnvManager::GetNumHumanState)
		.def("GetNumExoState",&EnvManager::GetNumExoState)
		.def("GetNumHumanAction",&EnvManager::GetNumHumanAction) 
		.def("GetNumExoAction",&EnvManager::GetNumExoAction)
		.def("GetSimulationHz",&EnvManager::GetSimulationHz)
		.def("GetControlHz",&EnvManager::GetControlHz)
		.def("GetNumSteps",&EnvManager::GetNumSteps)
		.def("UseMuscle",&EnvManager::UseMuscle)
		.def("Step",&EnvManager::Step)
		.def("Reset",&EnvManager::Reset)
		.def("IsEndOfEpisode",&EnvManager::IsEndOfEpisode)
		.def("GetReward",&EnvManager::GetReward) 
		.def("GetExoReward",&EnvManager::GetExoReward)  
		.def("GetHumanReward",&EnvManager::GetHumanReward)  
		.def("GetFullObservations",&EnvManager::GetFullObservations)  
		.def("Steps",&EnvManager::Steps)
		.def("StepsAtOnce",&EnvManager::StepsAtOnce)
		.def("Resets",&EnvManager::Resets)
		.def("IsEndOfEpisodes",&EnvManager::IsEndOfEpisodes)
		.def("GetStates",&EnvManager::GetStates)
		.def("GetExoStates",&EnvManager::GetExoStates) 
		.def("GetHumanStates",&EnvManager::GetHumanStates) 
		.def("SetHumanActions",&EnvManager::SetHumanActions)  
		.def("SetExoHumanActions",&EnvManager::SetExoHumanActions)  
		.def("GetHumanActions",&EnvManager::GetHumanActions)  
		.def("GetExoActions",&EnvManager::GetExoActions)
		.def("GetNumTotalMuscleRelatedDofs",&EnvManager::GetNumTotalMuscleRelatedDofs)
		.def("GetNumMuscles",&EnvManager::GetNumMuscles)
		.def("GetMuscleTorques",&EnvManager::GetMuscleTorques)
		.def("GetDesiredTorques",&EnvManager::GetDesiredTorques)
		.def("SetActivationLevels",&EnvManager::SetActivationLevels)
		.def("ComputeMuscleTuples",&EnvManager::ComputeMuscleTuples)
		.def("GetMuscleTuplesJtA",&EnvManager::GetMuscleTuplesJtA)
		.def("GetMuscleTuplesTauDes",&EnvManager::GetMuscleTuplesTauDes)
		.def("GetMuscleTuplesL",&EnvManager::GetMuscleTuplesL)
		.def("GetMuscleTuplesb",&EnvManager::GetMuscleTuplesb) 
		.def("UpdateTorqueBuffer",&EnvManager::UpdateTorqueBuffer)  
		.def("UpdateStateBuffers",&EnvManager::UpdateStateBuffers)  
		.def("UpdateExoActionBuffers",&EnvManager::UpdateExoActionBuffers)
		.def("UpdateHumanActionBuffers",&EnvManager::UpdateHumanActionBuffers); 
}