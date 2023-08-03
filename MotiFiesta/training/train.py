import sys
import time
import math
from collections import defaultdict

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from torchviz import make_dot

from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.utils.graph_utils import to_graphs
from MotiFiesta.utils.graph_utils import get_subgraphs

torch.autograd.set_detect_anomaly(True)

class Controller:
    def __init__(self, since_best_threshold=1):
        self.since_best_threshold = since_best_threshold
        self.modules = ['rec', 'mot']
        self.best_losses = {key: {'best_loss': float('nan'), 'since_best': 0}
                            for key in self.modules}
        pass

    def keep_going(self, key):
        """ Returns True if model should keep training, false otherwise."""

        if self.best_losses[key]['since_best'] > self.since_best_threshold:
            return False
        else:
            return True

    def update(self, losses):
        for key, l in losses.items():
            if l < self.best_losses[key]['best_loss']:
                self.best_losses[key]['best_loss'] = l
                self.best_losses[key]['since_best'] = 0
            elif not math.isnan(l):
                self.best_losses[key]['since_best'] += 1
            else:
                pass
        pass

    def state_dict(self):
        return {'since_best_threshold': self.since_best_threshold,
                'modules': self.modules,
                'best_losses': self.best_losses
                }

    def set_state(self, state_dict):
        self.since_best_threshold = state_dict['since_best_threshold']
        self.modules = state_dict['modules']
        self.best_losses = state_dict['best_losses']

def print_gradients(model):
    """
        Set the gradients to the embedding and the attributor networks.
        If True sets requires_grad to true for network parameters.
    """
    for param in model.named_parameters():
        name, p = param
        print(name, p, p.grad, p.requires_grad, p.shape)
    pass

def motif_train(model,
                train_loader,
                test_loader,
                model_name='default',
                estimator='kde',
                epochs=5,
                lam=1,
                beta=1,
                max_batches=-1,
                stop_epochs=30,
                volume=False,
                n_neighbors=30,
                hard_embed=False,
                epoch_start=0,
                optimizer=None,
                controller_state=None
                ):
    """motif_train.

    :param model: MotiFiesta model
    :param loader: Graph DataLoader
    :param null_loader: optional. loader containing 'null graphs'
    :param model_name: ID to save model under
    :param epochs: number of epochs to train
    :param lambda_rec: loss coefficient for embedding representation loss
    :param lambda_mot: loss coefficient for edge scores
    :param max_batches: if not -1, stop after given number of batches
    """
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    start_time = time.time()

    writer = SummaryWriter(f"logs/{model_name}")

    if controller_state is None:
        controller = Controller(since_best_threshold=stop_epochs,
                                )
    else:
        controller = Controller()
        controller.set_state(controller_state)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    mot_loss, rec_loss = [torch.tensor(float('nan'))] * 2
    done_training = False

    for epoch in range(epoch_start, epochs):
        if done_training:
            print("DONE TRAINING")
            break

        model.train()
        model.to(get_device())

        num_batches = len(train_loader)
        # only keep parameters that require grad

        rec_loss_tot, mot_loss_tot, = [0] * 2

        for batch_idx, batch in tqdm(enumerate(train_loader), total=num_batches):
            if batch_idx >= max_batches and max_batches > 0:
                break

            optimizer.zero_grad()

            graphs_pos = to_graphs(batch['pos'])
            graphs_neg = to_graphs(batch['neg'])

            batch_pos = batch['pos'].to(get_device())
            batch_neg = batch['neg'].to(get_device())

            x_pos, edge_index_pos = batch['pos'].x, batch['pos'].edge_index
            x_neg, edge_index_neg = batch['neg'].x, batch['neg'].edge_index


            # do main forward pass
            xx_pos, pp_pos, ee_pos,_, merge_info_pos, internals_pos = model(x_pos,
                                                                                 edge_index_pos,
                                                                                 batch_pos.batch
                                                                                 )
            xx_neg, pp_neg, ee_neg,_, merge_info_neg, internals_neg = model(x_neg,
                                                                                  edge_index_neg,
                                                                                  batch_neg.batch
                                                                                 )
            # d = make_dot(xx_pos[0], params=dict(model.named_parameters()))
            # d.render('Graph', view=True)
            loss = 0

            backward = False
            warmup_done = False

            if controller.keep_going('rec') and not hard_embed:
                rec_start = time.time()
                rec_loss = model.rec_loss(xx_pos,
                                          ee_pos,
                                          merge_info_pos['spotlights'],
                                          graphs_pos,
                                          batch_pos.batch,
                                          x_pos,
                                          internals_pos,
                                          draw=False
                                          )
                rec_loss_tot += rec_loss.item()
                rec_time = time.time() - rec_start
                backward = True
                loss += rec_loss
            else:
                warmup_done = True

            if controller.keep_going('mot') and warmup_done:
            # if True:
                mot_start = time.time()
                mot_loss = model.freq_loss(internals_pos,
                                           internals_neg,
                                           pp_pos,
                                           steps=model.steps,
                                           estimator=estimator,
                                           volume=volume,
                                           k=n_neighbors,
                                           lam=lam,
                                           beta=beta
                                           )


                loss += mot_loss
                mot_loss_tot += mot_loss.item()
                backward = True

                mot_time = time.time() - mot_start

            if backward:
                loss.backward()
                optimizer.step()
            else:
                done_training = True

        N = max_batches if max_batches > 0 else len(train_loader)

        losses = {'rec': rec_loss_tot / N,
                  'mot': mot_loss_tot / N,
                  }

        ## END OF BATCHES ##
        rec_loss_tot, mot_loss_tot, = [0] * 2

        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            if batch_idx >= max_batches and max_batches > 0:
                break

            model.eval()

            graphs_pos = to_graphs(batch['pos'])
            graphs_neg = to_graphs(batch['neg'])

            batch_pos = batch['pos'].to(get_device())
            batch_neg = batch['neg'].to(get_device())

            x_pos, edge_index_pos = batch['pos'].x, batch['pos'].edge_index
            x_neg, edge_index_neg = batch['neg'].x, batch['neg'].edge_index


            with torch.no_grad():
                # do main forward pass
                xx_pos, pp_pos, ee_pos, _, merge_info_pos, internals_pos = model(x_pos,
                                                                                        edge_index_pos,
                                                                                        batch_pos.batch
                                                                                        )
                xx_neg, pp_neg, ee_neg, _, merge_info_neg, internals_neg  = model(x_neg,
                                                                                            edge_index_neg,
                                                                                            batch_neg.batch
                                                                                            )
            warmup_done = False

            if controller.keep_going('rec'):
                rec_loss = model.rec_loss(xx_pos,
                                        ee_pos,
                                        merge_info_pos['spotlights'],
                                        graphs_pos,
                                        batch_pos.batch,
                                        x_pos,
                                        internals_pos
                                        )
            else:
                warmup_done = True

            mot_loss = torch.tensor(float('nan'))

            if warmup_done:
                mot_loss = model.freq_loss(internals_pos,
                                           internals_neg,
                                           pp_pos,
                                           steps=model.steps,
                                           estimator=estimator,
                                           volume=volume,
                                           k=n_neighbors,
                                           lam=lam,
                                           beta=beta
                                           )


            rec_loss_tot += rec_loss.item()
            mot_loss_tot += mot_loss.item()

        N = max_batches if max_batches > 0  else len(test_loader)

        test_losses = {'rec': rec_loss_tot / N,
                       'mot': mot_loss_tot / N,
                       }

        controller.update(test_losses)

        model.cpu()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'controller_state_dict': controller.state_dict()
        }, f'models/{model_name}/{model_name}.pth')

        loss_str = ' '.join([f'{k} train: {v:2f}' for k,v in losses.items()])
        test_loss_str = ' '.join([f'{k} test: {v:2f}' for k,v in test_losses.items()])
        model.to(get_device())
        time_elapsed = time.time() - start_time
        print(f"Train Epoch: {epoch+1} [{batch_idx +1}/{num_batches}]"\
              f"({100. * (batch_idx +1) / num_batches :.2f}%) {loss_str}"\
              f" {test_loss_str}"\
              f" Time: {time_elapsed:.2f}"
              )

        # tensorboard logging
        step = epoch * num_batches + batch_idx
        writer.add_scalar("Training loss", loss, step)
        for k,v in losses.items():
            writer.add_scalar(k, v, step)

        for k,v in test_losses.items():
            writer.add_scalar(k, v, step)

    model.cpu()
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'controller_state_dict': controller.state_dict()
    }, f'models/{model_name}/{model_name}.pth')
