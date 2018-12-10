from hyperopt import fmin, tpe, hp, Trials
import myvgg

def teste(params):
	epochs=params['epochs']
	MLPinput=params['MLPhidden']
	MLPhidden=params['MLPhidden']
	optimizer=params['optimizer']
	convtrain=params['convtrain']
	print('epochs=%d\nMLPinput=%d\nMLPhidden=%d\noptimizer=%s\nconvtrain=%d\nbatch_size=%d\n' % (epochs,MLPinput,MLPhidden,optimizer,convtrain,batch_size))
	return 1


batch_size = int(input("Digite o batch_size:"))

fspace = {
			'epochs': hp.choice('epochs',range(50,250)),
			'MLPinput': hp.choice('MLPinput',range(50,8192)),
			'MLPhidden': hp.choice('MLPhidden',range(50,8192)),
			'optimizer': hp.choice('optimizer',['sgd','prop']),
			'convtrain': hp.choice('convtrain',[11,17]),
			'batch_size': batch_size
		}

best = fmin(fn=myvgg.hyper,
    space= fspace,
    algo=tpe.suggest,
    max_evals=10)
print(best)