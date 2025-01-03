#ifndef __ENV_MANAGER_H__
#define __ENV_MANAGER_H__
#include "Environment.h"
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <utility>

namespace py = pybind11;

class EnvManager
{
public:
	EnvManager(std::string meta_file,int num_envs);

	int GetNumState();  
	int GetSimulationHz();
	int GetControlHz();
	int GetNumSteps();    

	int GetNumHumanAction();    
	int GetNumExoAction();   
	bool UseMuscle();   

	void Step(int id);
	void Reset(bool RSI,int id);
	bool IsEndOfEpisode(int id);  

	int GetNumExoState();  
	int GetNumHumanState();  

	/// reward of exo and human   
	double GetReward(int id);   
	double GetExoReward(int id);   
	double GetHumanReward(int id);     

	int GetNumFullObservation();  
	const Eigen::MatrixXd& GetFullObservations();   
	
	void Steps(int num, int donestep);
	void StepsAtOnce();   
	void Resets(bool RSI);  

	// state and action buffers   
	void UpdateTorqueBuffer();   
	void UpdateStateBuffers();    
	void UpdateExoActionBuffers(const Eigen::MatrixXd& humanactions);   
	void UpdateHumanActionBuffers(const Eigen::MatrixXd& humanactions);     

	const Eigen::VectorXd& IsEndOfEpisodes();
	const Eigen::MatrixXd& GetStates();   

	void SetHumanActions(const Eigen::MatrixXd& actions);   
	void SetExoHumanActions(const Eigen::MatrixXd& exoactions, const Eigen::MatrixXd& humanactions);   

	const Eigen::VectorXd& GetRewards();  
	const Eigen::VectorXd& GetExoRewards();  
	const Eigen::VectorXd& GetHumanRewards();  

	const Eigen::VectorXd& GetHumanActions(int id);        
	const Eigen::VectorXd& GetExoActions(int id);    

	const Eigen::MatrixXd& GetHumanStates();          
	const Eigen::MatrixXd& GetExoStates();     
	
	//For Muscle Transitions
	int GetNumTotalMuscleRelatedDofs(){return mEnvs[0]->GetNumTotalRelatedDofs();};
	int GetNumMuscles(){return mEnvs[0]->GetCharacter()->GetMuscles().size();}
	const Eigen::MatrixXd& GetMuscleTorques();
	const Eigen::MatrixXd& GetDesiredTorques();
	void SetActivationLevels(const Eigen::MatrixXd& activations);
	
	void ComputeMuscleTuples(); 
	const Eigen::MatrixXd& GetMuscleTuplesJtA();
	const Eigen::MatrixXd& GetMuscleTuplesTauDes();
	const Eigen::MatrixXd& GetMuscleTuplesL();
	const Eigen::MatrixXd& GetMuscleTuplesb();  

private:
	std::vector<MASS::Environment*> mEnvs;

	int mNumEnvs;
	int muscle_torque_cols;
	int tau_des_cols;

	Eigen::VectorXd mEoe;
	Eigen::VectorXd mRewards;
	Eigen::MatrixXd mStates;
	Eigen::MatrixXd mMuscleTorques;
	Eigen::MatrixXd mDesiredTorques;  

	Eigen::VectorXd mExoRewards;   
	Eigen::VectorXd mHumanRewards;    
	Eigen::MatrixXd mObservations;  

	Eigen::MatrixXd mExoStates;  
	Eigen::MatrixXd mHumanStates;  

	Eigen::MatrixXd mMuscleTuplesJtA;
	Eigen::MatrixXd mMuscleTuplesTauDes;
	Eigen::MatrixXd mMuscleTuplesL;
	Eigen::MatrixXd mMuscleTuplesb;  
};

#endif