from hyperopt import fmin, tpe, hp, Trials
import myvgg
import pickle

def teste(params):
	epochs=params['epochs']
	MLPinput=params['MLPhidden']
	MLPhidden=params['MLPhidden']
	optimizer=params['optimizer']
	convtrain=params['convtrain']
	print('epochs=%d\nMLPinput=%d\nMLPhidden=%d\noptimizer=%s\nconvtrain=%d\nbatch_size=%d\n' % (epochs,MLPinput,MLPhidden,optimizer,convtrain,batch_size))
	return 1


batch_size = int(input("Digite o batch_size:"))

def run_trials():
	trials_step = 1	# how many additional trials to do after loading saved trials. 1 = save after iteration
	max_trials = 1	# initial max_trials. put something small to not have to wait
	fspace = {
			'epochs': hp.choice('epochs',range(50,250)),
			'MLPinput': hp.choice('MLPinput',range(50,8192)),
			'MLPhidden': hp.choice('MLPhidden',range(50,8192)),
			'optimizer': hp.choice('optimizer',['sgd','prop']),
			'convtrain': hp.choice('convtrain',[11,17]),
			'batch_size': batch_size
		}
	try:	#try to load an already saved trials object, and increase the max
		trials = pickle.load(open("pesos/hyperopt", "rb"))
		#print("Found saved Trials! Loading...")
		max_trials = len(trials.trials) + trials_step
		print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
	except:  # create a new trials object and start searching
		trials = Trials()

	best = fmin(fn=myvgg.hyper,
				space= fspace,
				algo=tpe.suggest,
				max_evals=max_trials,
				trials=trials)

	print("Best:", best)
    
    # save the trials object
	with open("pesos/hyperopt", "wb") as f:
		pickle.dump(trials, f)

# loop indefinitely and stop whenever you like
while True:
	run_trials()