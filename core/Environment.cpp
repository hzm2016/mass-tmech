#include "Environment.h"
#include "DARTHelper.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
#include "dart/collision/bullet/bullet.hpp"
#include<cmath>
#include <algorithm> 
#include <random>

using namespace dart;
using namespace dart::simulation;
using namespace dart::dynamics;
using namespace MASS;

Environment::
Environment()
	:mControlHz(30),mSimulationHz(900),mWorld(std::make_shared<World>()),mUseMuscle(true),w_q(0.65),w_v(0.1),w_ee(0.15),w_com(0.1)
{
	history_buffer_human_state.setMaxLen(HISTORY_BUFFER_LEN); 
	history_buffer_exo_state.setMaxLen(HISTORY_BUFFER_LEN);  
	history_buffer_human_action.setMaxLen(HISTORY_BUFFER_LEN); 
	history_buffer_exo_action.setMaxLen(HISTORY_BUFFER_LEN);  
	history_buffer_human_torque.setMaxLen(HISTORY_BUFFER_LEN);
	history_buffer_exo_torque.setMaxLen(HISTORY_BUFFER_LEN);  
}

void
Environment::
Initialize(const std::string& meta_file,bool load_obj)
{
	std::ifstream ifs(meta_file);
	if(!(ifs.is_open()))
	{
		std::cout<<"Can't read file "<<meta_file<<std::endl;
		return;
	}
	std::string str;
	std::string index;
	std::stringstream ss;
	MASS::Character* character = new MASS::Character();
	while(!ifs.eof())
	{
		str.clear();
		index.clear();
		ss.clear();

		std::getline(ifs,str);
		ss.str(str);
		ss>>index;
		if(!index.compare("use_muscle"))
		{	
			std::string str2;
			ss>>str2;
			if(!str2.compare("true"))
				this->SetUseMuscle(true);
			else
				this->SetUseMuscle(false);
		}
		else if(!index.compare("con_hz")){
			int hz;
			ss>>hz;
			this->SetControlHz(hz);
		}
		else if(!index.compare("sim_hz")){
			int hz;
			ss>>hz;
			this->SetSimulationHz(hz);
		}
		else if(!index.compare("sim_hz")){
			int hz;
			ss>>hz;
			this->SetSimulationHz(hz);
		}
		else if(!index.compare("skel_file")){
			std::string str2;
			ss>>str2;

			character->LoadSkeleton(std::string(MASS_ROOT_DIR)+str2,load_obj);
		}
		else if(!index.compare("muscle_file")){
			std::string str2;
			ss>>str2;
			if(this->GetUseMuscle())
				character->LoadMuscles(std::string(MASS_ROOT_DIR)+str2);
		}
		else if(!index.compare("bvh_file")){
			std::string str2,str3;

			ss>>str2>>str3;
			bool cyclic = false;
			if(!str3.compare("true"))
				cyclic = true;
			character->LoadBVH(std::string(MASS_ROOT_DIR)+str2,cyclic);
		}
		else if(!index.compare("reward_param")){
			double a,b,c,d;
			ss>>a>>b>>c>>d;
			this->SetRewardParameters(a,b,c,d); 
		}
		else if(!index.compare("PD_param")){
			double kp;
			ss>>kp;
			this->SetPDParameters(kp);   
		}
	}
	ifs.close();  
	
	mUseExo = 1;   
	// double kp = 300.0;  
	character->SetPDParameters(mkp,sqrt(2*mkp));      
	this->SetCharacter(character);   
	this->SetGround(MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground.xml")));

	mNumExoActiveDof = 2;     
	upper_bound_p = M_PI;  
	lower_bound_p = -1 * M_PI; 
	upper_bound_v = M_PI; 
	lower_bound_v = -1 * M_PI;   

	this->Initialize();  

	std::cout << "NumExoActionDof: " << mNumExoActiveDof << std::endl; 
	std::cout << "kp: " << mkp << std::endl;      
}

void
Environment::
Initialize()
{
	if(mCharacter->GetSkeleton()==nullptr){
		std::cout<<"Initialize character First"<<std::endl;
		exit(0);  
	}   
	if(mCharacter->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="FreeJoint")
		mRootJointDof = 6;
	else if(mCharacter->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="PlanarJoint")
		mRootJointDof = 3;	
	else
		mRootJointDof = 0;   
	
	mNumActiveDof = mCharacter->GetSkeleton()->getNumDofs()-mRootJointDof;
	mNumHumanActiveDof = mNumActiveDof;  

	// usemuscle  
	if(mUseMuscle)
	{
		int num_total_related_dofs = 0;
		for(auto m : mCharacter->GetMuscles()){
			m->Update();
			num_total_related_dofs += m->GetNumRelatedDofs();
		}   
		mCurrentMuscleTuple.JtA = Eigen::VectorXd::Zero(num_total_related_dofs);
		mCurrentMuscleTuple.L = Eigen::VectorXd::Zero(mNumActiveDof*mCharacter->GetMuscles().size());
		mCurrentMuscleTuple.b = Eigen::VectorXd::Zero(mNumActiveDof);
		mCurrentMuscleTuple.tau_des = Eigen::VectorXd::Zero(mNumActiveDof);  
		mActivationLevels = Eigen::VectorXd::Zero(mCharacter->GetMuscles().size()); 
	}
	mWorld->setGravity(Eigen::Vector3d(0,-9.8,0.0));
	mWorld->setTimeStep(1.0/mSimulationHz);
	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
	mWorld->addSkeleton(mCharacter->GetSkeleton());
	mWorld->addSkeleton(mGround);  

	mAction = Eigen::VectorXd::Zero(mNumActiveDof); 

	// human action 
	mHumanAction = Eigen::VectorXd::Zero(mNumActiveDof);   
	mCurrentHumanAction = Eigen::VectorXd::Zero(mNumActiveDof);  
	mPrevHumanAction = Eigen::VectorXd::Zero(mNumActiveDof); 

	// exo action 
	mNumExoControlDof = mNumExoActiveDof;  
	mExoAction = Eigen::VectorXd::Zero(mNumExoControlDof);  
	mCurrentExoAction = Eigen::VectorXd::Zero(mNumExoControlDof);  
	mPrevExoAction = Eigen::VectorXd::Zero(mNumExoControlDof);   
	
	mDesiredTorque = Eigen::VectorXd::Zero(mNumHumanActiveDof);  
	mDesiredExoTorque = Eigen::VectorXd::Zero(mNumExoActiveDof);   

	// observation   
	Reset(false);  

	mNumState = GetState().rows();   

	/// human states  
	mNumHumanState = GetHumanState().rows();    

	/// exo states
	// mNumExoState = GetExoControlState().rows();      
	mNumExoState = GetExoTrueState().rows();    

	mNumExoControlState = 4 * 3;     
 
	std::cout << "NumState: " << mNumState << std::endl; 
	std::cout << "NumHumanState: " << mNumHumanState << std::endl;  
	std::cout << "NumExoState: " << mNumExoControlState << std::endl;  
	std::cout << "RootDof: " << mRootJointDof << std::endl;   
	std::cout << "HumanDof: " << mCharacter->GetHumandof() << std::endl;   
	std::cout << "NumBodyNodes: " << mCharacter->GetSkeleton()->getNumBodyNodes() << std::endl;   
}   

void
Environment::
Reset(bool RSI)
{
	mWorld->reset();

	mCharacter->GetSkeleton()->clearConstraintImpulses();
	mCharacter->GetSkeleton()->clearInternalForces();
	mCharacter->GetSkeleton()->clearExternalForces();
	
	double t = 0.0;

	if(RSI)
		t = dart::math::random(0.0,mCharacter->GetBVH()->GetMaxTime()*0.9);
	mWorld->setTime(t);
	mCharacter->Reset();

	mAction.setZero();

	mHumanAction.setZero();  
	mCurrentHumanAction.setZero();    
	mPrevHumanAction.setZero();     

	mExoAction.setZero();   
	mCurrentExoAction.setZero();   
	mPrevExoAction.setZero();     
	
	mDesiredExoTorque.setZero();  
	mDesiredTorque.setZero();  

	std::pair<Eigen::VectorXd,Eigen::VectorXd> pv = mCharacter->GetTargetPosAndVel(t,1.0/mControlHz);
	mTargetPositions = pv.first;   
	mTargetVelocities = pv.second;   

	mCharacter->GetSkeleton()->setPositions(mTargetPositions);
	mCharacter->GetSkeleton()->setVelocities(mTargetVelocities);
	mCharacter->GetSkeleton()->computeForwardKinematics(true,false,false);  

	randomized_latency = 0;  
	for(int i=0; i<HISTORY_BUFFER_LEN; i++)
	{
		history_buffer_human_state.push_back(this->GetHumanState());        
		history_buffer_exo_state.push_back(this->GetExoTrueState());         
		history_buffer_human_action.push_back(this->GetHumanAction());        
		history_buffer_exo_action.push_back(this->GetExoAction());      
		history_buffer_human_torque.push_back(this->GetDesiredTorques());      
		history_buffer_exo_torque.push_back(this->GetDesiredExoTorques());     
	}
}

void
Environment::
Step()
{	
	if(mUseMuscle)
	{
		int count = 0;
		for(auto muscle : mCharacter->GetMuscles())
		{
			muscle->activation = mActivationLevels[count++];
			muscle->Update();
			muscle->ApplyForceToBody();
		}
		if(mSimCount == mRandomSampleIndex)
		{
			auto& skel = mCharacter->GetSkeleton();
			auto& muscles = mCharacter->GetMuscles();

			int n = skel->getNumDofs();
			int m = muscles.size();
			Eigen::MatrixXd JtA = Eigen::MatrixXd::Zero(n,m);
			Eigen::VectorXd Jtp = Eigen::VectorXd::Zero(n);

			for(int i=0;i<muscles.size();i++)
			{
				auto muscle = muscles[i];
				// muscle->Update();
				Eigen::MatrixXd Jt = muscle->GetJacobianTranspose();
				auto Ap = muscle->GetForceJacobianAndPassive();

				JtA.block(0,i,n,1) = Jt*Ap.first;
				Jtp += Jt*Ap.second;
			}

			mCurrentMuscleTuple.JtA = GetMuscleTorques();
			Eigen::MatrixXd L = JtA.block(mRootJointDof,0,n-mRootJointDof,m);
			Eigen::VectorXd L_vectorized = Eigen::VectorXd((n-mRootJointDof)*m);
			for(int i=0;i<n-mRootJointDof;i++)
			{
				L_vectorized.segment(i*m, m) = L.row(i);
			}
			mCurrentMuscleTuple.L = L_vectorized;
			mCurrentMuscleTuple.b = Jtp.segment(mRootJointDof,n-mRootJointDof);
			mCurrentMuscleTuple.tau_des = mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
			mMuscleTuples.push_back(mCurrentMuscleTuple);
		}   
	}
	else
	{
		// without exo network 
		GetDesiredTorques();  
		// with exo network 
		// GetDesiredExoTorques();  
		mCharacter->GetSkeleton()->setForces(mDesiredTorque);   
		UpdateTorqueBuffer();       
	}

	mWorld->step();  

	// Eigen::VectorXd p_des = mTargetPositions;  
	// //p_des.tail(mAction.rows()) += mAction;  
	// mCharacter->GetSkeleton()->setPositions(p_des);  
	// mCharacter->GetSkeleton()->setVelocities(mTargetVelocities);   
	// mCharacter->GetSkeleton()->computeForwardKinematics(true,false,false);  
	// mWorld->setTime(mWorld->getTime()+mWorld->getTimeStep());  

	mSimCount++;
}

Eigen::VectorXd  
Environment::
GetDesiredTorques()  
{
	Eigen::VectorXd p_des = mTargetPositions;   
	p_des.tail(mNumHumanActiveDof) += mHumanAction;    

	mDesiredTorque = mCharacter->GetSPDForces(p_des);     
	return mDesiredTorque.tail(mNumHumanActiveDof);    
}   

Eigen::VectorXd  
Environment::
GetDesiredExoTorques()   
{
	Eigen::VectorXd p_des_human = mTargetPositions;  
	p_des_human.tail(mNumHumanActiveDof) += mHumanAction;   

	// only output reference position 
	Eigen::VectorXd p_des_exo = Eigen::VectorXd::Zero(mNumExoControlDof);     
	if (mNumExoControlDof == 4)
	{ 
		// output reference position and velocity  
		p_des_exo[0] = mTargetPositions[15];     
		p_des_exo[1] = mTargetPositions[6]; 
		p_des_exo[2] = mTargetVelocities[15];  
		p_des_exo[3] = mTargetVelocities[6];   
		p_des_exo += mExoAction;  
	}
	else
	{
		// only output reference position 
		p_des_exo[0] = mTargetPositions[15];     
		p_des_exo[1] = mTargetPositions[6];     
		p_des_exo += mExoAction;     
	}
	
	// std::cout << p_des_exo[0] << "," << p_des_exo[1] << std::endl;   
	std::pair<Eigen::VectorXd,Eigen::VectorXd> torque_results = mCharacter->GetSPDForces(p_des_human, p_des_exo);     
	mDesiredTorque = torque_results.first; 
	mDesiredExoTorque = torque_results.second;  

	return mDesiredExoTorque;   
}   

Eigen::VectorXd
Environment::
GetMuscleTorques()
{
	int index = 0;
	mCurrentMuscleTuple.JtA.setZero();
	for(auto muscle : mCharacter->GetMuscles())
	{
		muscle->Update();
		Eigen::VectorXd JtA_i = muscle->GetRelatedJtA();
		mCurrentMuscleTuple.JtA.segment(index,JtA_i.rows()) = JtA_i;
		index += JtA_i.rows();
	}
	
	return mCurrentMuscleTuple.JtA;
}

bool
Environment::
IsEndOfEpisode()
{
	bool isTerminal = false;
	
	Eigen::VectorXd p = mCharacter->GetSkeleton()->getPositions();
	Eigen::VectorXd v = mCharacter->GetSkeleton()->getVelocities();

	double root_y = mCharacter->GetSkeleton()->getBodyNode(0)->getTransform().translation()[1] - mGround->getRootBodyNode()->getCOM()[1];
	if(root_y<1.3)
		isTerminal =true;
	else if (dart::math::isNan(p) || dart::math::isNan(v))
		isTerminal =true;
	else if(mWorld->getTime()>10.0)
		isTerminal =true;
	
	return isTerminal;   
}

Eigen::VectorXd 
Environment::
GetState()
{
	auto& skel = mCharacter->GetSkeleton();
	dart::dynamics::BodyNode* root = skel->getBodyNode(0);
	int num_body_nodes = skel->getNumBodyNodes();
	Eigen::VectorXd p,v;  

	p.resize((num_body_nodes-1)*3);    
	v.resize((num_body_nodes)*3);    

	for(int i = 1;i<num_body_nodes;i++)   
	{
		p.segment<3>(3*(i-1)) = skel->getBodyNode(i)->getCOM(root);
		v.segment<3>(3*(i-1)) = skel->getBodyNode(i)->getCOMLinearVelocity();
	}
	
	v.tail<3>() = root->getCOMLinearVelocity();

	double t_phase = mCharacter->GetBVH()->GetMaxTime(); 
	double phi = std::fmod(mWorld->getTime(),t_phase)/t_phase;  

	p *= 0.8;
	v *= 0.2;

	Eigen::VectorXd state(p.rows()+v.rows()+1);

	state<<p,v,phi;
	return state;
}

void 
Environment::
SetAction(const Eigen::VectorXd& a)
{
	mAction = a*0.1;   

	double t = mWorld->getTime();   

	std::pair<Eigen::VectorXd,Eigen::VectorXd> pv = mCharacter->GetTargetPosAndVel(t,1.0/mControlHz);  
	mTargetPositions = pv.first;  
	mTargetVelocities = pv.second;   

	mSimCount = 0;
	mRandomSampleIndex = rand()%(mSimulationHz/mControlHz);
	mAverageActivationLevels.setZero();  
}

double 
Environment::
GetReward() 
{
	auto& skel = mCharacter->GetSkeleton();  

	Eigen::VectorXd cur_pos = skel->getPositions();  
	Eigen::VectorXd cur_vel = skel->getVelocities();     

	Eigen::VectorXd p_diff_all = skel->getPositionDifferences(mTargetPositions,cur_pos);
	Eigen::VectorXd v_diff_all = skel->getPositionDifferences(mTargetVelocities,cur_vel);

	Eigen::VectorXd p_diff = Eigen::VectorXd::Zero(skel->getNumDofs());  
	Eigen::VectorXd v_diff = Eigen::VectorXd::Zero(skel->getNumDofs());  

	const auto& bvh_map = mCharacter->GetBVH()->GetBVHMap();  

	for(auto ss : bvh_map)
	{
		auto joint = mCharacter->GetSkeleton()->getBodyNode(ss.first)->getParentJoint();
		int idx = joint->getIndexInSkeleton(0);
		if(joint->getType()=="FreeJoint")
			continue;
		else if(joint->getType()=="RevoluteJoint")
			p_diff[idx] = p_diff_all[idx];
		else if(joint->getType()=="BallJoint")
			p_diff.segment<3>(idx) = p_diff_all.segment<3>(idx);
	}

	auto ees = mCharacter->GetEndEffectors();  
	Eigen::VectorXd ee_diff(ees.size()*3);  
	Eigen::VectorXd com_diff;   

	// ee_diff  
	for(int i =0;i<ees.size();i++)
		ee_diff.segment<3>(i*3) = ees[i]->getCOM();
	
	// com   
	com_diff = skel->getCOM();

	skel->setPositions(mTargetPositions);
	skel->computeForwardKinematics(true,false,false);

	com_diff -= skel->getCOM();
	for(int i=0;i<ees.size();i++)
		ee_diff.segment<3>(i*3) -= ees[i]->getCOM()+com_diff;

	skel->setPositions(cur_pos);
	skel->computeForwardKinematics(true,false,false);

	double r_q = exp_of_squared(p_diff,2.0);
	double r_v = exp_of_squared(v_diff,0.1);
	double r_ee = exp_of_squared(ee_diff,40.0);
	double r_com = exp_of_squared(com_diff,10.0);

	double r = r_ee*(w_q*r_q + w_v*r_v);    

	// smooth the applied force
	Eigen::VectorXd torque = GetDesiredTorques().head(mCharacter->GetHumandof());   
	double r_torque = exp_of_squared(torque, 0.01);    
	// double r_torque = exp(-0.01*torque.squaredNorm());     

	return r;
}

Eigen::VectorXd 
Environment::   
GetHumanState()   
{
	auto& skel = mCharacter->GetSkeleton();     
	Eigen::VectorXd p_human, v_human;  
	p_human = skel->getPositions().tail(mNumHumanActiveDof);
	v_human = skel->getVelocities().tail(mNumHumanActiveDof);  
	Eigen::VectorXd p_cur_human, v_cur_human;
	p_cur_human = p_human;    
	v_cur_human = v_human/10.0;    
	Eigen::VectorXd human_state(p_cur_human.rows()+v_cur_human.rows());
	human_state << p_cur_human, v_cur_human;  
	return human_state;  
}

void 
Environment::
SetHumanAction(const Eigen::VectorXd& a)
{
	mPrevHumanAction = mCurrentHumanAction;   
	mCurrentHumanAction = a*0.1; 

	double t = mWorld->getTime();   

	std::pair<Eigen::VectorXd,Eigen::VectorXd> pv = mCharacter->GetTargetPosAndVel(t,1.0/mControlHz);  
	mTargetPositions = pv.first;  
	mTargetVelocities = pv.second;   

	mSimCount = 0;
	mRandomSampleIndex = rand()%(mSimulationHz/mControlHz);
	mAverageActivationLevels.setZero();  
}

double 
Environment::
GetHumanReward() 
{
	auto& skel = mCharacter->GetSkeleton();

	Eigen::VectorXd cur_pos = skel->getPositions();
	Eigen::VectorXd cur_vel = skel->getVelocities();

	Eigen::VectorXd p_diff_all = skel->getPositionDifferences(mTargetPositions,cur_pos);
	Eigen::VectorXd v_diff_all = skel->getPositionDifferences(mTargetVelocities,cur_vel);

	Eigen::VectorXd p_diff = Eigen::VectorXd::Zero(skel->getNumDofs());
	Eigen::VectorXd v_diff = Eigen::VectorXd::Zero(skel->getNumDofs());

	const auto& bvh_map = mCharacter->GetBVH()->GetBVHMap();  

	for(auto ss : bvh_map)
	{
		auto joint = mCharacter->GetSkeleton()->getBodyNode(ss.first)->getParentJoint();
		int idx = joint->getIndexInSkeleton(0);
		if(joint->getType()=="FreeJoint")
			continue;
		else if(joint->getType()=="RevoluteJoint")
			p_diff[idx] = p_diff_all[idx];
		else if(joint->getType()=="BallJoint")
			p_diff.segment<3>(idx) = p_diff_all.segment<3>(idx);
	}

	auto ees = mCharacter->GetEndEffectors();
	Eigen::VectorXd ee_diff(ees.size()*3);
	Eigen::VectorXd com_diff;  

	for(int i =0;i<ees.size();i++)
		ee_diff.segment<3>(i*3) = ees[i]->getCOM();
	com_diff = skel->getCOM();

	skel->setPositions(mTargetPositions);
	skel->computeForwardKinematics(true,false,false);

	com_diff -= skel->getCOM();
	for(int i=0;i<ees.size();i++)
		ee_diff.segment<3>(i*3) -= ees[i]->getCOM()+com_diff;

	skel->setPositions(cur_pos);  
	skel->computeForwardKinematics(true,false,false);  

	double r_q = exp_of_squared(p_diff,2.0);
	double r_v = exp_of_squared(v_diff,0.1);
	double r_ee = exp_of_squared(ee_diff,40.0);
	double r_com = exp_of_squared(com_diff,10.0);

	double r = r_ee*(w_q*r_q + w_v*r_v);   

	Eigen::VectorXd torque_human = GetDesiredTorques().tail(mNumHumanActiveDof);         
	double r_torque = exp_of_squared(torque_human, 0.01);       

	return r;  
}

void 
Environment::
SetExoAction(const Eigen::VectorXd& a)      
{
	mPrevExoAction = mCurrentExoAction;    
	mCurrentExoAction = a*1;  
	double t = mWorld->getTime();  
}

double
Environment::
GetExoReward()  
{
	// smooth torque or smooth joint angle  
   	Eigen::VectorXd torque_diff_exo = (history_buffer_exo_torque.get(HISTORY_BUFFER_LEN-1)-2*history_buffer_exo_torque.get(HISTORY_BUFFER_LEN-2)+history_buffer_exo_torque.get(HISTORY_BUFFER_LEN-3)); 
	double r_torque_smooth = exp_of_squared(torque_diff_exo, 15.0);     
	
	// // get exo torque  
 	Eigen::VectorXd torque_exo = GetDesiredExoTorques();    
    double r_torque_exo = exp_of_squared(mDesiredExoTorque, 0.01);      

	r_torque_exo = 0.001; 

	double r_human = GetHumanReward();     

	double r = r_torque_smooth + 0.01 * r_torque_exo;       

	// reward muscle efforts   
	double r_muscle = exp_of_squared(mActivationLevels, 0.01);     

	if (dart::math::isNan(r)){
		std::cout << "r_torque_smooth  "<< r_torque_smooth << " r_torque " << r_torque_exo << "r_muscle :" << r_muscle << "r_human" << r_human << std::endl;
	}   

	r = 0.1;  
	return r; 
}  

Eigen::VectorXd  
Environment::  
GetExoState() 
{  
	double dt = 1.0/mControlHz; 
	Eigen::VectorXd observation;   
	if((randomized_latency <= 0) || (history_buffer_exo_state.size() == 1)){
    	observation = history_buffer_exo_state.get(HISTORY_BUFFER_LEN-1);
	}
	else
	{
		int n_steps_ago = int(randomized_latency / dt);  
		if(n_steps_ago + 1 >= history_buffer_exo_state.size()){
			observation = history_buffer_exo_state.get(HISTORY_BUFFER_LEN-1);
		}
		else
		{
			double remaining_latency = randomized_latency - n_steps_ago * dt; 
			double blend_alpha = remaining_latency / dt; 
			observation = (
				(1.0 - blend_alpha) * history_buffer_exo_state.get(HISTORY_BUFFER_LEN - n_steps_ago - 1)
				+ blend_alpha * history_buffer_exo_state.get(HISTORY_BUFFER_LEN - n_steps_ago - 2)); 
		}
	}

    return observation;  
}

Eigen::VectorXd 
Environment::   
GetExoTrueState()    
{
	auto& skel = mCharacter->GetSkeleton();     
	// dart::dynamics::BodyNode* root = skel->getBodyNode(0);   // get root

	Eigen::VectorXd p_cur = skel->getPositions();
	Eigen::VectorXd v_cur = skel->getVelocities();

	Eigen::VectorXd p_save = Eigen::VectorXd::Zero(2);
	Eigen::VectorXd v_save = Eigen::VectorXd::Zero(2);  
	p_save[0] = p_cur[15];
	p_save[1] = p_cur[6]; 
	v_save[0] = v_cur[15];
	v_save[1] = v_cur[6];   

    Eigen::VectorXd state(p_save.rows()+v_save.rows()); //+tar_poses.rows());
	state<<p_save,v_save; //tar_poses;  
	return state;
}

void 
Environment:: 
UpdateStateBuffer()  
{
	history_buffer_human_state.push_back(this->GetHumanState());        
	history_buffer_exo_state.push_back(this->GetExoTrueState());         
}

void 
Environment:: 
UpdateExoActionBuffer(Eigen::VectorXd exoaction)  
{
	history_buffer_exo_action.push_back(exoaction);   
}

void 
Environment:: 
UpdateHumanActionBuffer(Eigen::VectorXd humanaction)
{
	history_buffer_human_action.push_back(humanaction); 
}

void  
Environment:: 
UpdateTorqueBuffer()  
{
	history_buffer_human_torque.push_back(mDesiredTorque);   
	history_buffer_exo_torque.push_back(mDesiredExoTorque);  
}

Eigen::VectorXd 
Environment:: 
GetExoControlState()
{
	// exo states    
	Eigen::MatrixXd states(mNumExoState, HISTORY_BUFFER_LEN);  
	for(int i=0; i<HISTORY_BUFFER_LEN; i++)  
		states.col(i) =  history_buffer_exo_state.get(i);    
	Eigen::VectorXd states_v = Eigen::Map<const Eigen::VectorXd>(states.data(), states.size());   

	// // exo actions  
	// Eigen::MatrixXd actions(mNumExoAction, HISTORY_BUFFER_LEN);      
	// for(int i=0; i<HISTORY_BUFFER_LEN; i++)   
	// 	actions.col(i) =  history_buffer_exo_action.get(i);    
	// Eigen::VectorXd actions_v = Eigen::Map<const Eigen::VectorXd>(actions.data(), actions.size());

	Eigen::VectorXd observation;   
	// observation.resize(states_v.rows()+actions_v.rows());   
	// observation << states_v,actions_v;  

	observation.resize(states_v.rows());     
	observation << states_v;   
    return observation;  
}

Eigen::VectorXd 
Environment::  
GetFullObservation()  
{  
	// exo states    
	Eigen::MatrixXd states(mNumExoState, HISTORY_BUFFER_LEN);
	for(int i=0; i<HISTORY_BUFFER_LEN; i++)
		states.col(i) =  history_buffer_exo_state.get(i); 
	Eigen::VectorXd states_v = Eigen::Map<const Eigen::VectorXd>(states.data(), states.size());   

	// exo actions  
	Eigen::MatrixXd actions(mNumExoAction, HISTORY_BUFFER_LEN);      
	for(int i=0; i<HISTORY_BUFFER_LEN; i++)   
		actions.col(i) =  history_buffer_exo_action.get(i);    
	Eigen::VectorXd actions_v = Eigen::Map<const Eigen::VectorXd>(actions.data(), actions.size());

	//get human states   
	Eigen::VectorXd humanstates_v = GetHumanState();    

	Eigen::VectorXd observation;   

	// get all states 
	if (mUseExo)
	{
		observation.resize(states_v.rows()+actions_v.rows()+humanstates_v.rows());
		observation << states_v,actions_v,humanstates_v;
	}
	else
	{
		observation.resize(humanstates_v.rows());
		observation << humanstates_v;
	}
	return observation;   
}

Eigen::VectorXd clamp(Eigen::VectorXd x, double lo, double hi)
{
	for(int i=0; i<x.rows(); i++)
	{
		x[i] = (x[i] < lo) ? lo : (hi < x[i]) ? hi : x[i];
	}
	return x; 
}

double clamp(double x, double lo, double hi)
{

	x = (x< lo) ? lo : (hi < x) ? hi : x;

	return x; 
}

void
Environment::  
ProcessAction(int substep_count, int num)
{
    double lerp = double(substep_count + 1) / num;     //substep_count: the step count should be between [0, num_action_repeat).
    mExoAction = mPrevExoAction + lerp * (mCurrentExoAction - mPrevExoAction);
	mHumanAction = mPrevHumanAction + lerp * (mCurrentHumanAction - mPrevHumanAction);
}

double 
Environment::  
exp_of_squared(const Eigen::VectorXd& vec,double w)
{
	return exp(-w*vec.squaredNorm());
}

double 
Environment::  
exp_of_squared(double val,double w)
{
	return exp(-w*val*val);
}