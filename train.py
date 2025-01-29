import torch
from earlystop import EarlyStopping
from test import test


import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle

import time

def train(
    model,
    config,
    train_loader,
    embedding_table,
    dist_index_dict,
    n_samples,
    criterion,
    diff_config,
    test_loader=None,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",

):

    early_stopping = EarlyStopping(config.patience, verbose=True, save_path=config.save_path, reverse=False)

    alpha = config.alpha
    alpha_decay = 0.8

    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config.epochs)
    p2 = int(0.9 * config.epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config.epochs):
        avg_loss = 0
        model.train()

        train_recover_loss, train_diff_loss, train_dist_loss = 0.0, 0.0, 0.0
        slices = train_loader.generate_batch(config.batch_size)

        timestart_per_epoch = time.time()

        with tqdm(slices, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, s in enumerate(it, start=1):

                optimizer.zero_grad()
                if config.dist_loss:
                #    batch = train_loader.get_slice(s)
                    batch = train_loader.get_slice(s,is_train=0)

                    inputs, targets, anchor_points, pos_points, neg_points, pos_weights = batch[:-5], batch[-5], \
                                                                                          batch[-4], batch[-3], \
                                                                                          batch[-2], batch[-1]
                #               scores = model(*inputs)
            #        diff_loss, discrete_celoss, tT_loss = model(inputs, n_samples, embedding_table, is_train=1)
                    #diff_loss, discrete_celoss, tT_loss = model(inputs, n_samples, embedding_table, is_train=0,validate=0) #original working
                    diff_loss, discrete_celoss, tT_loss = model(inputs,targets, n_samples, embedding_table, is_train=0, validate=0)

                    if epoch_no <100:
                        dist_loss = fn_dist_loss(model, anchor_points, pos_points, neg_points, pos_weights)
                        loss = diff_loss + discrete_celoss + tT_loss + alpha * dist_loss
                        train_dist_loss += alpha * dist_loss
                    else:
                        loss = diff_loss + discrete_celoss + tT_loss
                    train_recover_loss += discrete_celoss.item()
                    train_diff_loss += diff_loss.item()


                else:
            #        batch = train_loader.get_slice(s)
                    batch = train_loader.get_slice(s,is_train=0)
                    #inputs, targets = batch[:-1], batch[-1]
            #               scores = model(*inputs)
                    #inputs, targets = batch[:-1], batch[-1]
                    inputs = batch[:-1]
    #                optimizer.zero_grad()
                    targets = batch[-1]

                    '''diff_loss,scores,targets = model(inputs,n_samples,embedding_table)
                    #recover_loss = criterion(scores, targets - 3)
                    recover_loss = criterion(scores, targets-1)'''
                #    diff_loss, discrete_celoss, tT_loss = model(inputs, n_samples, embedding_table, is_train=1)
                    #diff_loss, discrete_celoss,tT_loss = model(inputs, n_samples, embedding_table,is_train=0,validate=0) #original working
                    diff_loss, discrete_celoss, tT_loss = model(inputs,targets, n_samples, embedding_table, is_train=0, validate=0)

                    

                    #loss = recover_loss + 1* diff_loss
                    loss = diff_loss + discrete_celoss + tT_loss

                    train_recover_loss += discrete_celoss.item()
                    train_diff_loss += diff_loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )

            alpha = alpha * alpha_decay

            training_time_per_epoch_before_lrstep = time.time() - timestart_per_epoch
                #recall1, recall3, recall5, recall10, dist1, dist3, dist5, dist10, map = test(model, valid_loader, config,
                #                                                                             embedding_table,
                #                                                                             dist_index_dict)
                #print("epoch: {}, recall@1: {:.4f}, recall@3: {:.4f}, recall@5: {:.4f}, recall@10: {:4f}".format(epoch_no,
                #                                                                                                 recall1,
                #                                                                                                 recall3,
                #                                                                                                 recall5,
                #                                                                                                 recall10))

            lr_scheduler.step()

        training_time_per_epoch_after_lrstep = time.time() - timestart_per_epoch

        print("epoch: {}, training loss: {:.4f}, recover loss: {:.4f}, diffusion loss: {:4f}, dist loss: {:4f}, training_time_per_epoch_before_lrstep: {:4f}, training_time_per_epoch_after_lrstep: {:4f}".format(
                                                                                                     epoch_no,
                                                                                                     avg_loss,
                                                                                                     train_recover_loss,
                                                                                                     train_diff_loss,
                                                                                                     train_dist_loss,
                                                                                                     training_time_per_epoch_before_lrstep,
                                                                                                      training_time_per_epoch_after_lrstep))

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
 #               with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
 #                   for batch_no, valid_batch in enumerate(it, start=1):
                
                timestart_validate_per_epoch = time.time()

                with tqdm(valid_loader.generate_batch(config.batch_size), mininterval=5.0, maxinterval=50.0) as it:

                    for batch_no, s in enumerate(it, start=1):

#                        loss = model(valid_batch, is_train=0)
                        batch = valid_loader.get_slice(s, is_train=0)
                        input, target = batch[:-1], batch[-1]

                        '''diff_loss,scores,_ = model(input,n_samples,embedding_table, is_train=0)
                        #recover_loss = criterion(scores, target-3)
                        recover_loss = criterion(scores, target-1)

                        loss = recover_loss + 1* diff_loss'''
                #        diff_loss, discrete_celoss,tT_loss = model(input, n_samples, embedding_table,is_train=0)
                        #diff_loss, discrete_celoss,tT_loss = model(input, n_samples, embedding_table,is_train=0,validate=1) #original working
                        diff_loss, discrete_celoss,tT_loss = model(input,target, n_samples, embedding_table,is_train=0,validate=1)

                        loss = diff_loss + discrete_celoss + tT_loss


                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

                time_for_validate_per_epoch = time.time() - timestart_validate_per_epoch
                print("time_for_validate_per_epoch: {:4f}".format(time_for_validate_per_epoch))

            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )



        recall1, recall3, recall5, recall10, dist1, dist3, dist5, dist10, map = test(model, test_loader, config,
                                                                                     embedding_table,
                                                                                     dist_index_dict,
                                                                                     diff_config["model"]["logits_mode"])

        time_after_inference_per_epoch = time.time() - timestart_per_epoch

        print("epoch: {}, recall@1: {:.4f}, recall@3: {:.4f}, recall@5: {:.4f}, recall@10: {:4f}, map: {:4f}".format(
            epoch_no, recall1, recall3, recall5,
            recall10,
            map))
        print("dist@1: {:.4f}, dist@3: {:.4f}, dist@5: {:.4f}, dist@10: {:4f}".format(dist1, dist3, dist5, dist10))

        print("time_after_inference_per_epoch: {:4f}".format(time_after_inference_per_epoch))


        early_stopping(recall1, model)
        if early_stopping.early_stop:
            print("Early Stopping!")
            break


#     if valid_loader is not None:
#         model.load_state_dict(torch.load(config.save_path))  #newly commented

    if foldername != "":
        torch.save(model.state_dict(), output_path)



def fn_dist_loss_original(model, anchor_points, pos_points, neg_points, pos_weights):
    anchor_emb = model.get_embedding(anchor_points)
    pos_emb = model.get_embedding(pos_points)
    neg_emb = model.get_embedding(neg_points)
    triple_loss = torch.nn.functional.triplet_margin_loss(anchor_emb, pos_emb, neg_emb, reduction="none")
    dist_loss = (triple_loss * pos_weights).sum()
    return dist_loss

def fn_dist_loss(model, anchor_points, pos_points, neg_points, pos_weights):
    anchor_emb = model.ID_embedding(anchor_points)
    pos_emb = model.ID_embedding(pos_points)
    neg_emb = model.ID_embedding(neg_points)
    triple_loss = torch.nn.functional.triplet_margin_loss(anchor_emb, pos_emb, neg_emb, reduction="none")
    dist_loss = (triple_loss * pos_weights).sum()
    return dist_loss



