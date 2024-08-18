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


disc = "Running bayesian hyperparameter optimization for benchmark"
usage = "usage: %prog [options]"
parser = OptionParser(usage=usage,description = disc)
parser.add_option("-c", action="store", type="int", dest="num_clusters", help="number of components for GMM",default=4)
parser.add_option("-n", action="store", type="string", dest="exp_name", help="name of the use_case")
parser.add_option("-p", action="store", type="string", dest="work_path", help="the path to save the outputs")
parser.add_option("-t", action="store", type="string", dest="opt_typ", help="chose optimizer RectifiedAdam or classic Adam",default='Adam')
parser.add_option("-m", action="store", type="int", dest="number_trials", help="number of trials to test for each GMM component in bayesian hyperparameter optimization. set number_trials to be 0, if you want to skip hyperparameter optimization", default=40)
parser.add_option("-v", action="store", type="string", dest="surv", help="whether you want to use survival analysis. True or False",default='True')


options, args = parser.parse_args()
num_cluster = options.num_clusters
exp_name = options.exp_name
work_path = os.path.abspath(options.work_path)
opt_typ = options.opt_typ
number_trials = options.number_trials
surv=options.surv

username = os.path.expanduser('~').split('/')[-1]

save_path = os.path.join(work_path, 'trained_model')

if not os.path.exists(save_path):
    # If the directory doesn't exist, create it
    os.makedirs(save_path)
    

temp_path=os.path.join(work_path, 'temp')

if not os.path.exists(work_path):
    # If the directory doesn't exist, create it
    os.makedirs(work_path)
    print(f"Directory '{work_path}' created.")

if not os.path.exists(temp_path):
    # If the directory doesn't exist, create it
    os.makedirs(temp_path)
    print(f"Directory '{temp_path}' created.")

def wait_for_jobs(num_job,time_out=99):
    #wait until all job finished
    check=0
    while(check==0):
        try:
            output = subprocess.check_output(["squeue", "-u",username])
            check=1
        except:
            pass
    output = [x for x in output.decode().split('\n') if 'TransVar' in x]
    running = len(output)
    while(running>num_job-1):
        time.sleep(60)
        check=0
        while(check==0):
            try:
                output = subprocess.check_output(["squeue", "-u",username])
                check=1
            except:
                pass
        output = [x for x in output.decode().split('\n') if 'TransVar' in x]
        running = len(output)
        for job in output:
            job_id = job.split()[0]
            job_status= job.split()[4]
            job_hour = job.split()[5].split('-')[-1].split(':')
            print(job_hour,len(job_hour),int(job_hour[0]),time_out)
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
    #you need modify these to set your own env
    
    if args['run_model']=='False':
        print('python3.9 '+script+' -c '+str(args['num_cluster'])+' -n '+str(args['exp_name'])+' -p '+str(args['work_path'])+' -t '+str(args['opt_typ'])+' -f '+str(args['outer_fold'])+' -w '+str(args['whole'])+' -s '+str(args['save_model'])+ ' -v '+str(args['surv']),file=out)
    elif args['run_model']=='True':
        print('python3.9 '+script+' -c '+str(args['num_cluster'])+' -n '+str(args['exp_name'])+' -p '+str(args['work_path'])+' -b '+str(args['benchmark'])+' -w '+str(args['whole'])+' -s '+str(args['nested']),file=out)
    
    out.close()
    output = subprocess.check_output(['sbatch',tmp_script])
    job_id = output.decode().rstrip().split(' ')[-1]

    #check whether the job has started
    check=glob.glob(os.path.join(temp_path, 'job_'+job_id+'_*'))
    while(not check):
        time.sleep(10)
        check=glob.glob(os.path.join(temp_path, 'job_'+job_id+'_*'))



if not num_cluster or not exp_name or not work_path or not opt_typ:
    parser.print_help()
    sys.exit()

if not os.path.exists(work_path):
    # If the directory doesn't exist, create it
    os.makedirs(work_path)


# Get the directory containing the script
script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path=os.path.join(script_path,'data_use_cases/'+exp_name)
temp_path=os.path.join(work_path, 'temp')

if not os.path.exists(temp_path):
    # If the directory doesn't exist, create it
    os.makedirs(temp_path)



K_folds=('34512','45123','51234','12345','23451')

for outer_fold in K_folds:
    if os.path.exists(os.path.join(save_path, 'model_weights_fold'+str(outer_fold)+'.index')):
        print('trained model already exists for fold '+str(outer_fold))
        continue
    try:
        study = optuna.load_study(study_name="train_num_cluster"+str(num_cluster), storage='sqlite:///'+work_path+'/train'+outer_fold+'.db')
        completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        num_completed_trials = len(completed_trials)
    except:
        num_completed_trials = 0
    print('start hyperparameter optimization,',num_completed_trials,'trials already finished',flush=True)
    #submit baysian hyperparamters optimization jobs

    print('submit jobs for testing '+str(num_cluster)+' GMM components',flush=True)
    if number_trials>num_completed_trials:
        for x in tqdm(range(number_trials-num_completed_trials)):
            wait_for_jobs(5)
            configs={}
            configs['num_cluster']=num_cluster
            configs['exp_name']=exp_name
            configs['work_path']=work_path
            configs['opt_typ']=opt_typ
            configs['outer_fold']=outer_fold
            configs['whole']='False'
            configs['run_model']='False'
            configs['save_model']='False'
            configs['surv']=str(surv)
            time.sleep(10)
            job_submit(str(os.path.join(script_path,'scripts/run-optuna-benchmark-inner.py')),configs)

    #wait until all job finished
    print('Wait until the bayesian hyperparameter optimization finished',flush=True)
    for x in tqdm(range(1)):
        wait_for_jobs(1)

    trial_idx=None
    max_value = -9999
    epoch=None
    study = optuna.load_study(study_name="train_num_cluster"+str(num_cluster), storage='sqlite:///'+work_path+'/train'+outer_fold+'.db')

    for t in study.trials:
        if t.state==optuna.trial.TrialState.COMPLETE and t.value > max_value:
            trial_idx=t.number
            max_value=t.value
            epoch=t.user_attrs['epoch']
    params = study.trials[int(trial_idx)].params
    params['num_cluster']=num_cluster
    params['epoch']=epoch
    params['survival']=surv

    outf=open(str(os.path.join(work_path,'params'+outer_fold+'.txt')),'w')
    for k in params:
        print(k,params[k],sep='\t',file=outf)
    outf.close()

    try:
        params={}
        for l in open(str(os.path.join(work_path,'params'+outer_fold+'.txt'))):
            l=l.rstrip()
            t=l.split('\t')
            params[t[0]]=t[1]
        num_cluster=params['num_cluster']
    except:
        print('where is your optimized hyperparameters?')
        sys.exit()
    
    print('Train the model',flush=True)

    configs={}
    configs['num_cluster']=num_cluster
    configs['exp_name']=exp_name
    configs['work_path']=work_path
    configs['opt_typ']=opt_typ
    configs['outer_fold']=outer_fold
    configs['whole']='False'
    configs['save_model']='True'
    configs['run_model']='False'
    configs['surv']=str(surv)
    job_submit(str(os.path.join(script_path,'scripts/run-optuna-benchmark-inner.py')),configs)

    print('Wait until the training of use case model finished',flush=True)
    for x in tqdm(range(1)):
        wait_for_jobs(1)

print('Evaluate the model',flush=True)        
configs={}
configs['num_cluster']=num_cluster
configs['exp_name']=exp_name
configs['work_path']=work_path
configs['benchmark']='True'
configs['run_model']='True'
configs['whole']='False'
configs['nested']='True'
job_submit(str(os.path.join(script_path,'scripts/run-eval.py')),configs)

print('All done.',flush=True)

if keep_temp!='True':
    shutil.rmtree(temp_path)
