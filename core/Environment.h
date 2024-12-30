#ifndef __MASS_ENVIRONMENT_H__
#define __MASS_ENVIRONMENT_H__
#include "dart/dart.hpp"
#include "Character.h"
#include "Muscle.h"
#include "Fixedeque.h" 

#include <queue>
#include <deque>

#define HISTORY_BUFFER_LEN 3 


namespace MASS
{
struct MuscleTuple
{
	Eigen::VectorXd JtA;
	Eigen::VectorXd L;
	Eigen::VectorXd b;
	Eigen::VectorXd tau_des;
};
class Environment
{
public:
	Environment();

	void SetUseMuscle(bool use_muscle){mUseMuscle = use_muscle;}
	void SetControlHz(int con_hz) {mControlHz = con_hz;}
	void SetSimulationHz(int sim_hz) {mSimulationHz = sim_hz;}  

	void SetCharacter(Character* character) {mCharacter = character;}
	void SetGround(const dart::dynamics::SkeletonPtr& ground) {mGround = ground;}

	void SetRewardParameters(double w_q,double w_v,double w_ee,double w_com){this->w_q = w_q;this->w_v = w_v;this->w_ee = w_ee;this->w_com = w_com;}
	void SetPDParameters(double kp) {mkp = kp;}
 	void Initialize();
	void Initialize(const std::string& meta_file,bool load_obj = false);  

	int GetNumFullObservation(){return mNumFullObservation;}   

public:
	void Step();
	void Reset(bool RSI = true);  
	bool IsEndOfEpisode();  

	double exp_of_squared(const Eigen::VectorXd& vec,double w); 
	double exp_of_squared(double val,double w); 

	void ProcessAction(int j, int num);   

	Eigen::VectorXd GetState();  
	void SetAction(const Eigen::VectorXd& a);   
	double GetReward();    

	// human  
	Eigen::VectorXd GetHumanState();     
	void SetHumanAction(const Eigen::VectorXd& a);  
	double GetHumanReward();  

	// exo  
	Eigen::VectorXd GetExoState();    
	void SetExoAction(const Eigen::VectorXd& a);    
	double GetExoReward();   
	Eigen::VectorXd GetExoTrueState();    
	Eigen::VectorXd GetExoControlState();     

	// update buffers    
	void UpdateStateBuffer();    
	void UpdateTorqueBuffer();   
	void UpdateExoActionBuffer(Eigen::VectorXd exoaction);       
	void UpdateHumanActionBuffer(Eigen::VectorXd humanaction);   
  
	// get total observation 
	Eigen::VectorXd GetFullObservation();  

	// get desired force 
	Eigen::VectorXd GetDesiredTorques();   
	Eigen::VectorXd GetDesiredExoTorques();     
	Eigen::VectorXd GetMuscleTorques();    

	// get actions  
	Eigen::VectorXd GetExoAction() {return mExoAction;}    
	Eigen::VectorXd GetHumanAction() {return mHumanAction;}      

	const dart::simulation::WorldPtr& GetWorld(){return mWorld;}
	Character* GetCharacter(){return mCharacter;}
	const dart::dynamics::SkeletonPtr& GetGround(){return mGround;}
	int GetControlHz(){return mControlHz;}
	int GetSimulationHz(){return mSimulationHz;}
	int GetNumTotalRelatedDofs(){return mCurrentMuscleTuple.JtA.rows();}
	std::vector<MuscleTuple>& GetMuscleTuples(){return mMuscleTuples;};  

	int GetNumState(){return mNumState;}   
	int GetNumExoState(){return mNumExoControlState;}    
	int GetNumHumanState(){return mNumHumanState;}  
	int GetNumHumanAction(){return mNumActiveDof;}   
	int GetNumExoAction(){return mNumExoControlDof;}   

	int GetNumSteps(){return mSimulationHz/mControlHz;}
	
	const Eigen::VectorXd& GetActivationLevels(){return mActivationLevels;}
	const Eigen::VectorXd& GetAverageActivationLevels(){return mAverageActivationLevels;}
	void SetActivationLevels(const Eigen::VectorXd& a){mActivationLevels = a;}
	bool GetUseMuscle(){return mUseMuscle;}    

private:
	dart::simulation::WorldPtr mWorld;
	int mControlHz,mSimulationHz;
	bool mUseMuscle;
	Character* mCharacter;
	dart::dynamics::SkeletonPtr mGround;
	
	// state buffer   
	FixedQueue<Eigen::VectorXd> history_buffer_human_state;  
	FixedQueue<Eigen::VectorXd> history_buffer_exo_state;   

	// action buffer 
	FixedQueue<Eigen::VectorXd> history_buffer_exo_action;   
	FixedQueue<Eigen::VectorXd> history_buffer_human_action;    

	// torque buffer
	FixedQueue<Eigen::VectorXd> history_buffer_exo_torque;    
	FixedQueue<Eigen::VectorXd> history_buffer_human_torque;    

	// human 
	Eigen::VectorXd mAction;
	Eigen::VectorXd mTargetPositions,mTargetVelocities;  

	// exo   
	Eigen::VectorXd mExoAction;  
	Eigen::VectorXd mCurrentExoAction, mPrevExoAction;   

	// human    
	Eigen::VectorXd mHumanAction;  
	Eigen::VectorXd mCurrentHumanAction, mPrevHumanAction;      

	int mNumState;
	int mNumActiveDof;   
	int mRootJointDof;   

	int mNumFullObservation;  

	int mNumHumanState;    
	int mNumHumanAction;    

	int mNumExoState;    
	int mNumExoAction;    

	int mNumExoControlState;    

	int mNumHumanActiveDof;   
	int mNumExoActiveDof;   
	int mNumExoControlDof;    
	Eigen::VectorXd mDesiredExoTorque;    

	int mUseExo;   

	double randomized_latency;  

	Eigen::VectorXd mActivationLevels;   
	Eigen::VectorXd mAverageActivationLevels;   
	Eigen::VectorXd mDesiredTorque;   
	std::vector<MuscleTuple> mMuscleTuples;   
	MuscleTuple mCurrentMuscleTuple;   

	int mSimCount;   
	int mRandomSampleIndex;    

	double w_q,w_v,w_ee,w_com;  

	double mkp;  

	double mkp_human,mkp_exo;  

	double upper_bound_p;   
	double lower_bound_p; 
	double upper_bound_v;  
	double lower_bound_v;  
};
};

#endif