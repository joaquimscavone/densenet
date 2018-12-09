from hyperopt import fmin, tpe, hp, Trials
import myvgg




def teste(params):
	epochs=params['epochs']
	MLPinput=params['MLPhidden']
	MLPhidden=params['MLPhidden']
	optimizer=params['optimizer']
	mark=params['mark']
	print('epochs=%d\nMLPinput=%d\nMLPhidden=%d\noptimizer=%s\nmark=%d' % (epochs,MLPinput,MLPhidden,optimizer,mark))
	return 1







fspace = {
			'epochs': hp.choice('epochs',range(50,250)),
			'MLPinput': hp.choice('MLPinput',range(50,8192)),
			'MLPhidden': hp.choice('MLPhidden',range(50,8192)),
			'optimizer': hp.choice('optimizer',['sgd','prop']),
			'mark': hp.choice('mark',[11,17])
		}

best = fmin(fn=myvgg.hyper,
    space= fspace,
    algo=tpe.suggest,
    max_evals=10)
print(best)