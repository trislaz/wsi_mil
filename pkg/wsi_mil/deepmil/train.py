from .arguments import get_arguments
from .dataloader import Dataset_handler
from .models import DeepMIL
import numpy as np
# For the sklearn warnings
import warnings
warnings.filterwarnings('always')

def writes_metrics(writer, to_write, epoch):
    """writes_metrics.
    Writes the validation metrics (and the train loss) in a Tensorboard Writer.

    :param writer: Tensorboard Writer
    :param to_write: dict, scalars to write.
    :param epoch: time step.
    """
    for key in to_write:
        if type(to_write[key]) == dict:
            writer.add_scalars(key, to_write[key], epoch)
        else:
            writer.add_scalar(key, to_write[key], epoch)

def train(model, dataloader):
    model.network.train()
    mean_loss = []
    epobatch = 1/len(dataloader) # How many epochs per batch ?
    for input_batch, target_batch in dataloader:
        model.counter["batch"] += 1
        model.counter['epoch'] += epobatch
        [scheduler.step(model.counter['epoch']) for scheduler in model.schedulers]
        loss = model.optimize_parameters(input_batch, target_batch)
        mean_loss.append(loss)
    model.mean_train_loss = np.mean(mean_loss)
    print('train_loss: {}'.format(np.mean(mean_loss)))

def val(model, dataloader):
    model.network.eval()
    mean_loss = []
    for input_batch, target_batch in dataloader:
        target_batch = target_batch.to(model.device)
        loss = model.evaluate(input_batch, target_batch)
        mean_loss.append(loss)
    model.mean_val_loss = np.mean(mean_loss)
    to_write = model.flush_val_metrics()
    writes_metrics(model.writer, to_write, model.counter['epoch']) 
    state = model.make_state()
    print('mean val loss {}'.format(np.mean(mean_loss)))
    model.update_learning_rate(model.mean_val_loss)
    model.early_stopping(model.args.sgn_metric * to_write[model.args.ref_metric], state)

def main(raw_args=None):
    args = get_arguments(raw_args=raw_args, train=True)
    model = DeepMIL(args=args, with_data=True)
    model.get_summary_writer()
    while model.counter['epoch'] < args.epochs:
        print("Epochs {}".format(round(model.counter['epoch'])))
        train(model=model, dataloader=model.train_loader)
        if args.use_val:
            val(model=model, dataloader=model.val_loader)
        if model.early_stopping.early_stop:
            break
        if not args.use_val:
            torch.save(model.make_state(), 'model_best.pt.tar')
    model.writer.close()

