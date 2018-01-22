import datetime

import torch
from torch.autograd import Variable


def train(model, loss_fn, optimizer, loader_train, loader_val, num_epochs=1,
          acc_every=1, eval_result={}, save=False, lr_schedular=None):
    gpu_dtype = torch.cuda.FloatTensor
    print_every = 100
    train_loss_dict = {}
    val_loss_dict = {}
    train_acc = {}
    val_acc = {}
    LR = {}
    save_version = 0
    try:
        for epoch in range(num_epochs):
            print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
            model.train()
            train_loss = 0
            for t, (x, y) in enumerate(loader_train):
                x_var = Variable(x.type(gpu_dtype))
                y_var = Variable(y.type(gpu_dtype).long())

                # scores = model(x_var)
                # google net
                scores, score_aux1, score_aux2 = model(x_var)
                loss = loss_fn(scores, y_var)
                loss_aux1 = loss_fn(score_aux1, y_var)
                loss_aux2 = loss_fn(score_aux2, y_var)
                if (t + 1) % print_every == 0:
                    print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))
                train_loss += loss.data[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # epoch training loss
            train_loss_dict[epoch] = train_loss/(t+1)
            if epoch % acc_every == 0:
                lr = optimizer.param_groups[0]['lr']
                loss, acc = check_accuracy(model, loader_val, loss_fn)
                val_loss_dict[epoch] = loss
                val_acc[epoch] = acc
                LR[epoch] = lr
                print '-----> [{3}], epoch: {0}, train_loss: {1}, val_loss: {2}'.format(epoch, train_loss/(t+1), loss,
                                                                                        datetime.datetime.now().strftime('%H:%M:%S'))
                if acc >= 0.91 and save:
                    torch.save(model.state_dict(),
                               'training_model_version_{0}_{1}.pkl'.format(save_version, acc))
                    save_version += 1
                # LR schedular
                # if LR_schedular and epoch%40==39:
                #     change_LR(optimizer, 0.3)
                if lr_schedular:
                    lr_schedular.step(loss)
                    if lr_schedular.restart:
                        optimizer = lr_schedular.optim
                    if lr_schedular.reload:
                        check_accuracy(model, loader_val, loss_fn)
            if epoch % 100 == -1:
                loss, acc = check_accuracy(model, loader_train, loss_fn)
                train_acc[epoch] = acc
    except Exception, e:
        print e

    eval_result['train_loss'] = train_loss_dict
    eval_result['val_loss'] = val_loss_dict
    eval_result['train_acc'] = train_acc
    eval_result['val_acc'] = val_acc
    eval_result['lr'] = LR
    if save:
        model.eval()
        torch.save(model.state_dict(), 'model_final_version.pkl')
    return eval_result


def check_accuracy(model, loader, loss_fn):
    gpu_dtype = torch.cuda.FloatTensor
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    loss = 0.0
    count = 0
    for x, y in loader:
        x_var = Variable(x.type(gpu_dtype), volatile=True)
        y_var = Variable(y.type(gpu_dtype).long())

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
        loss += loss_fn(scores, y_var)
        count += 1

    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return loss.data.cpu().numpy()[0] / count, acc