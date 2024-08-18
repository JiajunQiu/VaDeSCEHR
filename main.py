import subprocess
import time
import os
from optparse import OptionParser
import matplotlib.pyplot as plt
import glob
import sys
import optuna
from tqdm import tqdm
import shutil
import random
import string

    
def wait_for_jobs(num_job,time_out=99):
    #wait until all job finished
    output = subprocess.check_output(["squeue", "-u",username])
    output = [x for x in output.decode().split('\n') if 'VaDeSCEHR' in x]
    running = len(output)
    while(running>num_job-1):
        time.sleep(60)
        output = subprocess.check_output(["squeue", "-u",username])
        output = [x for x in output.decode().split('\n') if 'VaDeSCEHR' in x]
        running = len(output)
        for job in output:
            job_id = job.split()[0]
            job_status= job.split()[4]
            job_hour = job.split()[5].split('-')[-1].split(':')
            if job_status=='R' and (len(job_hour)==3 and int(job_hour[0])>=time_out):
                os.system('scancel '+job_id)

def job_submit(script,args):
    barcode = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=4))
    tmp_script=temp_path+'/run-optuna-'+barcode+'.sh'
    out=open(tmp_script,'w')
    print('#!/bin/bash',file=out)
    print('#SBATCH -J VaDeSCEHR',file=out)  
    if 'evaluation' in args or args['run_model']=='True':
        print('#SBATCH -p gpu',file=out)
        print('#SBATCH -N 1',file=out)             
        print('#SBATCH --gres gpu:1',file=out)
        print('#SBATCH --mem-per-gpu 48G',file=out)
    else:
        print('#SBATCH --mem 48G',file=out)
    
    print('#SBATCH -t 29-12:00',file=out)            
    print('#SBATCH -o '+temp_path+'/job_%A_%a.log',file=out)       
    print('#SBATCH -e '+temp_path+'/job_%A_%a.log',file=out)      

    if args['scenario']=='use_case':
        if args['run_model']=='False':
            print('python3.9 '+script+' -c '+str(args['num_cluster'])+' -n '+str(args['exp_name'])+' -p '+str(args['work_path'])+' -t '+str(args['opt_typ'])+' -b '+str(args['benchmark'])+' -s '+str(args['save_model'])+' -w '+str(args['whole']),file=out)
        elif args['run_model']=='True':
            print('python3.9 '+script+' -c '+str(args['num_cluster'])+' -n '+str(args['exp_name'])+' -p '+str(args['work_path'])+' -b '+str(args['benchmark']),file=out)
    else:
        print('python3.9 '+script+' -p '+str(args['work_path'])+' -d '+str(args['data_fil'])+' -e '+str(args['evaluation'])+' -s '+str(args['save_model']),file=out) 

    
    out.close()
    if args['scenario']=='use_case':
        output = subprocess.check_output(['sbatch',tmp_script])
        job_id = output.decode().rstrip().split(' ')[-1]

        #check whether the job has started
        check=glob.glob(os.path.join(temp_path, 'job_'+job_id+'_*'))
        while(not check):
            time.sleep(10)
            check=glob.glob(os.path.join(temp_path, 'job_'+job_id+'_*'))
    else:
        os.system('sh '+tmp_script)


#Commandline parsing
disc = "Running VaDeSCEHR"
usage = "usage: %prog [options]"
parser = OptionParser(usage=usage,description = disc)
parser.add_option("-s", action="store", type="string", dest="scenario", help="run VaDeSCEHR in which scenario: use_case or simulation")
parser.add_option("-n", action="store", type="string", dest="exp_name", help="name of the use_case or the path to the simulation data set")
parser.add_option("-m", action="store", type="int", dest="number_trials", help="number of trials to test for each GMM component in bayesian hyperparameter optimization. set number_trials to be 0, if you want to skip hyperparameter optimization", default=40)
parser.add_option("-c", action="store", type="int", dest="number_components", help="max number of components for GMM to test in bayesian hyperparameter optimization or the known number of cluster for the data",default=4)
parser.add_option("-p", action="store", type="string", dest="work_path", help="the path to save the outputs")
parser.add_option("-b", action="store", type="string", dest="benchmark", help="whether you run on a benchmark data, aka whether you know the labels. True or False",default='False')
parser.add_option("-r", action="store", type="string", dest="run_model", help="run the previously trained model (True/False), exclusive to -m and -c")
parser.add_option("-t", action="store", type="string", dest="opt_typ", help="chose optimizer RectifiedAdam or classic Adam",default='Adam')
parser.add_option("-k", action="store", type="string", dest="keep_temp", help="whether to keep temp dir (True/False), default is False",default='False')

options, args = parser.parse_args()

scenario=options.scenario
exp_name=options.exp_name
number_trials = options.number_trials
number_components = options.number_components

work_path = options.work_path
benchmark = options.benchmark

run_model = options.run_model
opt_typ = options.opt_typ

keep_temp = options.keep_temp

username = os.path.expanduser('~').split('/')[-1]
script_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()

if not work_path or not scenario or not exp_name or ((not run_model) and (not isinstance(number_trials, (int, float)) or not isinstance(number_components, (int, float)))):
    parser.print_help()
    sys.exit()

temp_path=os.path.join(work_path, 'temp')

if not os.path.exists(work_path):
    # If the directory doesn't exist, create it
    os.makedirs(work_path)
    print(f"Directory '{work_path}' created.")

if not os.path.exists(temp_path):
    # If the directory doesn't exist, create it
    os.makedirs(temp_path)
    print(f"Directory '{temp_path}' created.")


if run_model=='True':
    print('Evaluate the model',flush=True)
    if scenario=='simulation':
        configs={}
        configs['data_fil']=exp_name
        configs['scenario']=scenario
        configs['run_model']=run_model
        configs['work_path']=work_path
        configs['evaluation']='True'
        configs['save_model']='False'
        job_submit(str(os.path.join(script_path,'scripts/run-simulation.py')),configs)
    elif scenario=='use_case':
        try:
            params={}
            for l in open(os.path.join(script_path,'data_use_cases/'+exp_name+'/params.txt')):
                l=l.rstrip()
                t=l.split('\t')
                params[t[0]]=t[1]
            num_cluster=params['num_cluster']
        except:
            print('where is your optimized hyperparameters?')
            sys.exit()
        
        configs={}
        configs['num_cluster']=num_cluster
        configs['scenario']=scenario
        configs['exp_name']=exp_name
        configs['run_model']=run_model
        configs['work_path']=work_path
        configs['benchmark']=benchmark
        configs['run_model']='True'
        job_submit(str(os.path.join(script_path,'scripts/run-eval.py')),configs)
        
    for x in tqdm(range(1)):
        wait_for_jobs(1)

    print('All done.You can find your results at '+str(work_path),flush=True)

    if keep_temp !='True':
        shutil.rmtree(temp_path)
    sys.exit()


if scenario=='use_case' and number_trials!=0:
    print('start hyperparameter optimization',flush=True)
    #submit baysian hyperparamters optimization jobs
    for cls in range(2,number_components+1):
        print('submit jobs for testing '+str(cls)+' GMM components',flush=True)
        for x in tqdm(range(number_trials)):
            wait_for_jobs(5)
            configs={}
            configs['num_cluster']=cls
            configs['exp_name']=exp_name
            configs['scenario']=scenario
            configs['opt_typ']=opt_typ
            configs['run_model']=run_model
            configs['work_path']=work_path
            configs['benchmark']=benchmark
            configs['save_model']='False'
            configs['run_model']='False'
            configs['whole']='False'
            job_submit(str(os.path.join(script_path,'scripts/run-optuna.py')),configs)

    #wait until all job finished
    print('Wait until the bayesian hyperparameter optimization finished',flush=True)
    for x in tqdm(range(1)):
        wait_for_jobs(1)

    trial_bic_merge=[]
    custom_metrics_merge=[]
    label_merge=[]
    for cls in range(2,number_components+1):
        study = optuna.load_study(study_name="train_num_cluster"+str(cls), storage='sqlite:///'+work_path+'/train.db')

        # Extract the trial numbers and custom metric values
        trial_numbers = [t.number for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]

        custom_metrics = [t.user_attrs['CI'] for t in study.trials  if t.state==optuna.trial.TrialState.COMPLETE]
        custom_metrics_merge=custom_metrics_merge+custom_metrics

        trial_bic = [t.value for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
        trial_bic = [x if x is not None else 9999 for x in trial_bic]
        trial_bic_merge = trial_bic_merge+trial_bic

        label_merge=label_merge+[str(cls)+'@'+str(x) for x in trial_numbers]


    custom_metrics_merge = [(x-min(custom_metrics_merge))/(max(custom_metrics_merge)-min(custom_metrics_merge)) for x in custom_metrics_merge] 
    trial_bic_merge = [1-((x-min(trial_bic_merge))/(max(trial_bic_merge)-min(trial_bic_merge))) for x in trial_bic_merge]


    # Plot the optimization history with the my_custom_metric attribute
    fig, ax = plt.subplots(figsize=(10, 10))
    dist=[]
    for idx,val in enumerate(trial_bic_merge):
        dist.append(trial_bic_merge[idx]**2+custom_metrics_merge[idx]**2)

    # Plot the scatter plot
    plt.scatter(trial_bic_merge, custom_metrics_merge)
    plt.xlabel('BIC of GMM', fontsize=30)
    plt.ylabel('Concordance index (C-index)', fontsize=30)
    #plt.ylim(0, 1)
    max_dist_index = dist.index(max(dist)) # get the index of the maximum distance point

    plt.scatter(trial_bic_merge[max_dist_index], custom_metrics_merge[max_dist_index], c='red', s=100, edgecolors='none')
    plt.tick_params(axis='both', labelsize=30)
    output_file=os.path.join(work_path, 'hyperparameter_opt.tiff')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')

    num_cluster,trial_idx = label_merge[dist.index(max(dist))].split('@')

    study_ori = optuna.load_study(study_name='train_num_cluster'+str(num_cluster), storage='sqlite:///'+work_path+'/train.db')
    params = study_ori.trials[int(trial_idx)].params
    params['num_cluster']=num_cluster
    outf=open(str(os.path.join(script_path,'data_use_cases/'+exp_name+'/params.txt')),'w')
    for k in params:
        print(k,params[k],sep='\t',file=outf)
    outf.close()



if scenario=='use_case':
    try:
        params={}
        for l in open(str(os.path.join(script_path,'data_use_cases/'+exp_name+'/params.txt'))):
            l=l.rstrip()
            t=l.split('\t')
            params[t[0]]=t[1]
        num_cluster=params['num_cluster']
    except:
        print('where is your optimized hyperparameters?')
        sys.exit()
    
print('Train the model',flush=True)

if scenario=='simulation':
    configs={}
    configs['data_fil']=exp_name
    configs['scenario']=scenario
    configs['evaluation']='False'
    configs['save_model']='True'
    configs['work_path']=work_path
    configs['run_model']=run_model
    job_submit(str(os.path.join(script_path,'scripts/run-simulation.py')),configs)
    print('Wait until the training of simulation model finished',flush=True)
    for x in tqdm(range(1)):
        wait_for_jobs(1)
elif scenario=='use_case':
    configs={}
    configs['num_cluster']=num_cluster
    configs['scenario']=scenario
    configs['exp_name']=exp_name
    configs['opt_typ']=opt_typ
    configs['work_path']=work_path
    configs['run_model']=run_model
    configs['benchmark']=benchmark
    configs['save_model']='True'
    configs['whole']='False'
    job_submit(str(os.path.join(script_path,'scripts/run-optuna.py')),configs)
    configs['whole']='True'
    job_submit(str(os.path.join(script_path,'scripts/run-optuna.py')),configs)
    print('Wait until the training of use case model finished',flush=True)
    for x in tqdm(range(1)):
        wait_for_jobs(1)

print('Evaluate the model',flush=True)        
if scenario=='simulation':
    configs={}
    configs['data_fil']=exp_name
    configs['work_path']=work_path
    configs['scenario']=scenario
    configs['run_model']=run_model
    configs['evaluation']='True'
    configs['save_model']='False'
    job_submit(str(os.path.join(script_path,'scripts/run-simulation.py')),configs)
elif scenario=='use_case':
    configs={}
    configs['num_cluster']=num_cluster
    configs['scenario']=scenario
    configs['exp_name']=exp_name
    configs['run_model']=run_model
    configs['work_path']=work_path
    configs['benchmark']=benchmark
    configs['run_model']='True'
    job_submit(str(os.path.join(script_path,'scripts/run-eval.py')),configs)

print('All done.',flush=True)

if keep_temp!='True':
    shutil.rmtree(temp_path)
