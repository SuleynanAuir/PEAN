import torch
import sys
import time
import os
import csv
from datetime import datetime
import copy
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
sys.path.append('../')
sys.path.append('./')
from interfaces import base
from utils.metrics import get_str_list
from utils.util import str_filt
from utils.visualization import plot_training_curves, plot_comparison_table, plot_diffusion_features, plot_prediction_examples


class TextSR(base.TextBase):

    def train(self):
        cfg = self.config.TRAIN
        print("Loading training data...")
        _, train_loader = self.get_train_data()
        print(f"Training data loaded: {len(train_loader)} batches")
        _, val_loader_list = self.get_val_data()
        print("Validation data loaded")
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        print("Generator initialized")

        parseq = self.PARSeq_init()
        parseq.eval()
        for p in parseq.parameters():
            p.requires_grad = False
        print("PARSeq loaded and set to eval mode")
        self.diffusion = self.init_diffusion_model()
        print("Diffusion model initialized")

        optimizer_G = self.optimizer_init([model, self.diffusion.netG])
        
        # Add adaptive learning rate scheduler
        from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
        # Use ReduceLROnPlateau for adaptive LR based on validation accuracy
        scheduler = ReduceLROnPlateau(optimizer_G, mode='max', factor=0.5, patience=3, 
                                     verbose=True, min_lr=1e-6)
        print("Adaptive learning rate scheduler initialized (ReduceLROnPlateau)")
        print("Optimizer initialized")  

        if not os.path.exists(cfg.ckpt_dir):
            os.makedirs(cfg.ckpt_dir)
            
        best_history_acc = dict(easy_aster=0, medium_aster=0, hard_aster=0,
                                easy_crnn=0, medium_crnn=0, hard_crnn=0,
                                easy_moran=0, medium_moran=0, hard_moran=0)        
        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc_aster = 0
        best_acc_crnn = 0
        best_acc_moran = 0
        converge_list = []
        
        # Initialize metrics history for visualization
        self.metrics_history = {
            'iterations': [],
            'loss': [],
            'psnr_easy': [], 'psnr_medium': [], 'psnr_hard': [],
            'ssim_easy': [], 'ssim_medium': [], 'ssim_hard': [],
            'acc_aster_easy': [], 'acc_aster_medium': [], 'acc_aster_hard': [],
            'learning_rate': []
        }
        
        log_path = os.path.join(cfg.ckpt_dir, "log.csv")

        print('='*110)
        display = True
        ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
        
        for epoch in range(cfg.epochs):
            for j, data in enumerate(train_loader):
                if display:
                    start = time.time()
                    display = False
                model.train()

                for p in model.parameters():
                    p.requires_grad = True
                iters = len(train_loader) * epoch + j + 1 
                images_hr, images_lr, labels, label_vecs, weighted_mask, weighted_tics = data
                
                text_label = label_vecs
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

                # Calculate text_len early to avoid UnboundLocalError
                text_sum = text_label.sum(1).squeeze(1)
                text_pos = (text_sum > 0).float().sum(1)
                text_len = text_pos.reshape(-1)

                # Batch process PARSeq inference for HR images
                batch_size = images_hr.shape[0]
                parseq_inputs = torch.cat([self.parse_parseq_data(images_hr[i, :3, :, :]) for i in range(batch_size)], dim=0)
                prob_str_hr = parseq(parseq_inputs, max_length=25).softmax(-1)

                if not self.args.pre_training:
                    # Batch process PARSeq inference for LR images
                    batch_size = images_lr.shape[0]
                    parseq_inputs = torch.cat([self.parse_parseq_data(images_lr[i, :3, :, :]) for i in range(batch_size)], dim=0)
                    prob_str_lr = parseq(parseq_inputs, max_length=25).softmax(-1)
                    predicted_length = torch.ones(prob_str_lr.shape[0]) * prob_str_lr.shape[1]

                    data_diff = {"HR":prob_str_hr, "SR":prob_str_lr, "weighted_mask": weighted_mask, "predicted_length": predicted_length, "text_len": text_len}
                    self.diffusion.feed_data(data_diff)
                    loss_diff, label_vecs_final = self.diffusion.process()
                    label_vecs_final = label_vecs_final.to(self.device)
                else:
                    label_vecs_final = prob_str_hr
                    loss_diff = 0

                images_sr, logits = model(images_lr, label_vecs_final)

                if self.args.pre_training:
                    predicted_length = torch.ones(logits.shape[1]) * logits.shape[0]

                loss_im = 0
                loss_im = loss_im + image_crit(images_sr, images_hr, labels) * 100
                aux_rec_loss = 0
                loss_aux_rec_module = 0
                aux_rec_loss += ctc_loss(logits.log_softmax(2),
                                         weighted_mask.long().to(self.device),
                                         predicted_length.long().to(self.device),
                                         text_len.long())
                aux_rec_loss_each = (aux_rec_loss * weighted_tics.float().to(self.device))
                aux_rec_loss_each = aux_rec_loss_each.mean()
                loss_aux_rec_module += aux_rec_loss_each
                loss_im = loss_im + loss_aux_rec_module
                loss_im = loss_im + loss_diff

                optimizer_G.zero_grad()
                loss_im.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer_G.step()
                
                # Record metrics for visualization
                if not hasattr(self, 'metrics_history'):
                    self.metrics_history = {
                        'iterations': [], 'loss': [], 'learning_rate': [],
                        'psnr_easy': [], 'psnr_medium': [], 'psnr_hard': [],
                        'ssim_easy': [], 'ssim_medium': [], 'ssim_hard': [],
                        'acc_aster_easy': [], 'acc_aster_medium': [], 'acc_aster_hard': []
                    }
                
                self.metrics_history['iterations'].append(iters)
                self.metrics_history['loss'].append(float(loss_im.data))
                self.metrics_history['learning_rate'].append(optimizer_G.param_groups[0]['lr'])

                if iters % cfg.displayInterval == 0:
                    end = time.time()
                    duration = end - start
                    display = True
                    print('[{}] | '
                          'Epoch: [{}][{} / {}] | '
                          'Loss: {} | Duration: {}s'
                            .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  epoch, j + 1, len(train_loader),
                                  float(loss_im.data), duration))
                    print('-'*110)

                if iters % cfg.VAL.valInterval == 0:
                    print('='*110)
                    current_acc_dict = {}
                    all_examples = {'correct': [], 'incorrect': []}
                    all_diffusion_states = []
                    
                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].replace('\\', '/').split('/')[-1]
                        print('evaling %s' % data_name)
                        # Collect examples only from first validation dataset
                        collect_examples = (k == 0)
                        metrics_dict = self.eval(model, val_loader, epoch, parseq, collect_examples=collect_examples)
                        converge_list.append({'iterator': iters,
                                              'acc_aster': metrics_dict['accuracy_aster'],
                                              'acc_crnn': metrics_dict['accuracy_crnn'],
                                              'acc_moran': metrics_dict['accuracy_moran'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        acc_aster = metrics_dict['accuracy_aster']
                        acc_crnn = metrics_dict['accuracy_crnn']
                        acc_moran = metrics_dict['accuracy_moran']
                        current_acc_dict[data_name+'_aster'] = float(acc_aster)
                        current_acc_dict[data_name+'_crnn'] = float(acc_crnn)
                        current_acc_dict[data_name+'_moran'] = float(acc_moran)
                        
                        # Record validation metrics for visualization
                        self.metrics_history[f'psnr_{data_name}'].append(metrics_dict['psnr_avg'])
                        self.metrics_history[f'ssim_{data_name}'].append(metrics_dict['ssim_avg'])
                        self.metrics_history[f'acc_aster_{data_name}'].append(float(acc_aster))
                        
                        # Collect examples and diffusion states from first dataset
                        if collect_examples:
                            if 'examples' in metrics_dict:
                                all_examples['correct'].extend(metrics_dict['examples']['correct'])
                                all_examples['incorrect'].extend(metrics_dict['examples']['incorrect'])
                            if 'diffusion_states' in metrics_dict:
                                all_diffusion_states.extend(metrics_dict['diffusion_states'])
                        
                        if acc_aster > best_history_acc[data_name+'_aster']:
                            best_history_acc[data_name+'_aster'] = float(acc_aster)
                            best_history_acc['epoch'] = epoch
                            best_model_info = {'accuracy_aster': float(acc_aster), 'psnr': metrics_dict['psnr_avg'], 'ssim': metrics_dict['ssim_avg']}
                            print('best_%s = %.2f%% (A New Record)' % (data_name+'_aster', best_history_acc[data_name+'_aster'] * 100))
                            self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list, data_name+'_aster')
                            self.diffusion.save_network(epoch, j, "aster")
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow([epoch, data_name+'_aster', metrics_dict['accuracy_aster'], metrics_dict['psnr_avg'], metrics_dict['ssim_avg'], "best_{}".format(data_name+'_aster')])
                        else:
                            print('best_%s_aster = %.2f%%' % (data_name, best_history_acc[data_name+'_aster'] * 100))
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow([epoch, data_name+'_aster', metrics_dict['accuracy_aster'], metrics_dict['psnr_avg'], metrics_dict['ssim_avg']])

                        if acc_crnn > best_history_acc[data_name+'_crnn']:
                            best_history_acc[data_name+'_crnn'] = float(acc_crnn)
                            best_history_acc['epoch'] = epoch
                            best_model_info = {'accuracy_crnn': float(acc_crnn), 'psnr': metrics_dict['psnr_avg'], 'ssim': metrics_dict['ssim_avg']}
                            print('best_%s = %.2f%% (A New Record)' % (data_name+'_crnn', best_history_acc[data_name+'_crnn'] * 100))
                            self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list, data_name+'_crnn')
                            self.diffusion.save_network(epoch, j, "crnn")
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow([epoch, data_name+'_crnn', metrics_dict['accuracy_crnn'], metrics_dict['psnr_avg'], metrics_dict['ssim_avg'], "best_{}".format(data_name+'_crnn')])
                        else:
                            print('best_%s_crnn = %.2f%%' % (data_name, best_history_acc[data_name+'_crnn'] * 100))
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow([epoch, data_name+'_crnn', metrics_dict['accuracy_crnn'], metrics_dict['psnr_avg'], metrics_dict['ssim_avg']])
                        
                        if acc_moran > best_history_acc[data_name+'_moran']:
                            best_history_acc[data_name+'_moran'] = float(acc_moran)
                            best_history_acc['epoch'] = epoch
                            best_model_info = {'accuracy_moran': float(acc_moran), 'psnr': metrics_dict['psnr_avg'], 'ssim': metrics_dict['ssim_avg']}
                            print('best_%s = %.2f%% (A New Record)' % (data_name+'_moran', best_history_acc[data_name+'_moran'] * 100))
                            self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list, data_name+'_moran')
                            self.diffusion.save_network(epoch, j, "moran")
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow([epoch, data_name+'_moran', metrics_dict['accuracy_moran'], metrics_dict['psnr_avg'], metrics_dict['ssim_avg'], "best_{}".format(data_name+'_moran')])
                        else:
                            print('best_%s_moran = %.2f%%' % (data_name, best_history_acc[data_name+'_moran'] * 100))
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow([epoch, data_name+'_moran', metrics_dict['accuracy_moran'], metrics_dict['psnr_avg'], metrics_dict['ssim_avg']])
                        print('-'*110)
                    
                    # Update learning rate scheduler based on average accuracy
                    avg_acc = (current_acc_dict['easy_aster'] + current_acc_dict['medium_aster'] + current_acc_dict['hard_aster']) / 3
                    scheduler.step(avg_acc)
                    
                    # Generate visualization plots after each validation
                    try:
                        # Create a unique subfolder for this validation
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        vis_subfolder = f'iter_{iters:07d}_epoch_{epoch:03d}_{timestamp}'
                        vis_dir = os.path.join(cfg.ckpt_dir, 'visualizations', vis_subfolder)
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        print(f'Saving visualizations to: {vis_dir}')
                        
                        plot_training_curves(self.metrics_history, save_path=os.path.join(vis_dir, 'training_curves.png'))
                        plot_comparison_table(self.metrics_history, save_path=os.path.join(vis_dir, 'metrics_table.png'))
                        
                        # Generate diffusion feature visualization if states were collected
                        if len(all_diffusion_states) > 0:
                            # Use simulated timesteps for visualization
                            timesteps = [1000, 500, 0][:len(all_diffusion_states)]
                            plot_diffusion_features(all_diffusion_states[:3], timesteps, 
                                                  save_path=os.path.join(vis_dir, 'diffusion_process.png'))
                        
                        # Generate prediction examples visualization
                        if len(all_examples['correct']) > 0 or len(all_examples['incorrect']) > 0:
                            plot_prediction_examples(all_examples, 
                                                   save_path=os.path.join(vis_dir, 'prediction_examples.png'))
                        
                        # Also save to 'latest' folder for easy access
                        latest_dir = os.path.join(cfg.ckpt_dir, 'visualizations', 'latest')
                        os.makedirs(latest_dir, exist_ok=True)
                        plot_training_curves(self.metrics_history, save_path=os.path.join(latest_dir, 'training_curves.png'))
                        plot_comparison_table(self.metrics_history, save_path=os.path.join(latest_dir, 'metrics_table.png'))
                        if len(all_diffusion_states) > 0:
                            timesteps = [1000, 500, 0][:len(all_diffusion_states)]
                            plot_diffusion_features(all_diffusion_states[:3], timesteps, 
                                                  save_path=os.path.join(latest_dir, 'diffusion_process.png'))
                        if len(all_examples['correct']) > 0 or len(all_examples['incorrect']) > 0:
                            plot_prediction_examples(all_examples, 
                                                   save_path=os.path.join(latest_dir, 'prediction_examples.png'))
                        
                        print(f'✓ Visualizations saved successfully!')
                    except Exception as e:
                        print(f'Warning: Failed to generate plots: {e}')
                    
                    if (current_acc_dict['easy_aster'] * 1619 + current_acc_dict['medium_aster'] * 1411 + current_acc_dict['hard_aster'] * 1343) / (1619 + 1411 + 1343) > best_acc_aster:       # 三个测试集识别准确率的和大于历史最好时
                        best_acc_aster = (current_acc_dict['easy_aster'] * 1619 + current_acc_dict['medium_aster'] * 1411 + current_acc_dict['hard_aster'] * 1343) / (1619 + 1411 + 1343)
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        print('saving best model for aster')
                        self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list, 'sum_aster')
                        self.diffusion.save_network(epoch, j, "sum_aster")
                        with open(log_path, "a+", newline="") as out:
                            writer = csv.writer(out)
                            writer.writerow([epoch, "", "", "", "", "", "best_sum_aster"])
                    if (current_acc_dict['easy_crnn'] * 1619 + current_acc_dict['medium_crnn'] * 1411 + current_acc_dict['hard_crnn'] * 1343) / (1619 + 1411 + 1343) > best_acc_crnn:       # 三个测试集识别准确率的和大于历史最好时
                        best_acc_crnn = (current_acc_dict['easy_crnn'] * 1619 + current_acc_dict['medium_crnn'] * 1411 + current_acc_dict['hard_crnn'] * 1343) / (1619 + 1411 + 1343)
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        print('saving best model for crnn')
                        self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list, 'sum_crnn')
                        self.diffusion.save_network(epoch, j, "sum_crnn")
                        with open(log_path, "a+", newline="") as out:
                            writer = csv.writer(out)
                            writer.writerow([epoch, "", "", "", "", "", "best_sum_crnn"])
                    if (current_acc_dict['easy_moran'] * 1619 + current_acc_dict['medium_moran'] * 1411 + current_acc_dict['hard_moran'] * 1343) / (1619 + 1411 + 1343) > best_acc_moran:       # 三个测试集识别准确率的和大于历史最好时
                        best_acc_moran = (current_acc_dict['easy_moran'] * 1619 + current_acc_dict['medium_moran'] * 1411 + current_acc_dict['hard_moran'] * 1343) / (1619 + 1411 + 1343)
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        print('saving best model for moran')
                        self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list, 'sum_moran')
                        self.diffusion.save_network(epoch, j, "sum_moran")
                        with open(log_path, "a+", newline="") as out:
                            writer = csv.writer(out)
                            writer.writerow([epoch, "", "", "", "", "", "best_sum_moran"])
                    print('='*110)

                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, False, converge_list)
                    self.diffusion.save_network("ckpt", "ckpt", "ckpt")

    def eval(self, model, val_loader, index, parseq, collect_examples=False):
        for p in model.parameters():
            p.requires_grad = False

        aster, aster_info = self.Aster_init()
        aster.eval()
        for p in aster.parameters():
            p.requires_grad = False
        
        # Only load CRNN if model file exists
        crnn = None
        crnn_path = self.config.TRAIN.VAL.crnn_pretrained
        if os.path.exists(crnn_path):
            crnn = self.CRNN_init()
            crnn.eval()
            for p in crnn.parameters():
                p.requires_grad = False
        else:
            print(f"Warning: CRNN model not found at {crnn_path}, skipping CRNN evaluation")
        
        # Only load MORAN if model file exists
        moran = None
        moran_path = self.config.TRAIN.VAL.moran_pretrained
        if os.path.exists(moran_path):
            moran = self.MORAN_init()
            moran.eval()
            for p in moran.parameters():
                p.requires_grad = False
        # else: silently skip MORAN if not available

        model.eval()
        n_correct_aster = 0
        n_correct_crnn = 0
        n_correct_moran = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        
        # For collecting examples
        correct_examples = []
        incorrect_examples = []
        diffusion_states = []  # For collecting diffusion features
        for _, data in (enumerate(val_loader)):
            images_hr, images_lr, label_strs, _ = data
            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            prob_str_lr = []
            for i in range(images_lr.shape[0]):
                parseq_input = self.parse_parseq_data(images_lr[i, :3, :, :])
                parseq_output = parseq(parseq_input)
                pred = parseq_output.softmax(-1)
                prob_str_lr.append(pred)
            prob_str_lr = torch.cat(prob_str_lr, dim=0)

            if not self.args.pre_training:
                label_vecs_final_new = None
                for j in range(val_batch_size):
                    data_diff = {"SR": prob_str_lr[j, :, :].unsqueeze(0)}
                    self.diffusion.feed_data(data_diff)
                    self.diffusion.test()
                    visuals = self.diffusion.get_current_visuals()
                    prior = visuals['SR']
                    if label_vecs_final_new is None:
                        label_vecs_final_new = prior
                    else:
                        label_vecs_final_new = torch.concat([label_vecs_final_new, prior], dim=0)
                label_vecs_final_new = label_vecs_final_new.to(self.device)
            else:
                prob_str_hr = []
                for j in range(images_hr.shape[0]):
                    parseq_input = self.parse_parseq_data(images_hr[j, :3, :, :])
                    parseq_output = parseq(parseq_input)
                    pred = parseq_output.softmax(-1)
                    prob_str_hr.append(pred)
                label_vecs_final_new = torch.cat(prob_str_hr, dim=0)

            images_sr, _ = model(images_lr, label_vecs_final_new)

            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

            aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
            aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
            aster_output_lr = aster(aster_dict_lr)
            aster_output_sr = aster(aster_dict_sr)
            pred_rec_lr = aster_output_lr['output']['pred_rec']
            pred_rec_sr = aster_output_sr['output']['pred_rec']
            pred_str_lr_aster, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            pred_str_sr_aster, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

            # CRNN prediction (only if model is available)
            if crnn is not None:
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output, _ = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr_crnn = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
            else:
                pred_str_sr_crnn = [''] * val_batch_size
            
            # MORAN prediction (only if model is available)
            if moran is not None:
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True, debug=True)
                preds, _ = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr_moran = [pred.split('$')[0] for pred in sim_preds]
            else:
                pred_str_sr_moran = [''] * val_batch_size
                
            for pred, target in zip(pred_str_sr_aster, label_strs):
                if pred == str_filt(target, 'lower'):
                    n_correct_aster += 1
            if crnn is not None:
                for pred, target in zip(pred_str_sr_crnn, label_strs):
                    if pred == str_filt(target, 'lower'):
                        n_correct_crnn += 1
            if moran is not None:
                for pred, target in zip(pred_str_sr_moran, label_strs):
                    if pred == str_filt(target, 'lower'):
                        n_correct_moran += 1

            # Collect examples for visualization (only if requested and limited samples)
            if collect_examples and (len(correct_examples) < 10 or len(incorrect_examples) < 10):
                for i in range(val_batch_size):
                    pred = pred_str_sr_aster[i]
                    label = str_filt(label_strs[i], 'lower')
                    is_correct = (pred == label)
                    
                    example = {
                        'lr': images_lr[i].cpu(),
                        'sr': images_sr[i].cpu(),
                        'hr': images_hr[i].cpu(),
                        'pred': pred,
                        'label': label
                    }
                    
                    if is_correct and len(correct_examples) < 10:
                        correct_examples.append(example)
                    elif not is_correct and len(incorrect_examples) < 10:
                        incorrect_examples.append(example)
                    
                    # Stop collecting if we have enough examples
                    if len(correct_examples) >= 10 and len(incorrect_examples) >= 10:
                        break
            
            # Collect diffusion features (sample one from batch if requested)
            if collect_examples and len(diffusion_states) < 6 and not self.args.pre_training:
                # Sample diffusion state from label_vecs_final_new (text prior after diffusion)
                if label_vecs_final_new is not None and len(diffusion_states) == 0:
                    # Store the final diffusion output as one of the states
                    diffusion_states.append(label_vecs_final_new[0].cpu())

            sum_images += val_batch_size
            torch.cuda.empty_cache()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        print('[{}] | '
              'PSNR {:.2f} | SSIM {:.4f}'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(psnr_avg), float(ssim_avg)))
        print('save display images')

        self.tripple_display(images_lr, images_sr, images_hr, pred_str_lr_aster, pred_str_sr_aster, label_strs, index)
        accuracy_aster = round(n_correct_aster / sum_images, 4)
        accuracy_crnn = round(n_correct_crnn / sum_images, 4)
        accuracy_moran = round(n_correct_moran / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        print('aster_accuray: %.2f%% | crnn_accuray: %.2f%% | moran_accuray: %.2f%% ' % (accuracy_aster * 100, accuracy_crnn * 100, accuracy_moran * 100))
        metric_dict['accuracy_aster'] = accuracy_aster
        metric_dict['accuracy_crnn'] = accuracy_crnn
        metric_dict['accuracy_moran'] = accuracy_moran
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg
        
        # Add collected examples to metrics if requested
        if collect_examples:
            metric_dict['examples'] = {
                'correct': correct_examples,
                'incorrect': incorrect_examples
            }
            metric_dict['diffusion_states'] = diffusion_states
        
        return metric_dict

    def test(self):
        model_dict = self.generator_init()
        model, _ = model_dict['model'], model_dict['crit']
        if not self.args.pre_training:
            self.diffusion = self.init_diffusion_model()
            self.diffusion.netG.eval()
            for p in self.diffusion.netG.parameters():
                p.requires_grad = False
        _, test_loader = self.get_test_data(self.test_data_dir)
        data_name = self.args.test_data_dir.split('\\')[-1]
        print('evaling %s' % data_name)
        parseq = self.PARSeq_init()
        for p in parseq.parameters():
            p.requires_grad = False
        parseq.eval()
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()

        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        n_correct = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        current_acc_dict = {data_name: 0}
        time_begin = time.time()
        print('='*110)
        
        for i, data in (enumerate(test_loader)):
            images_hr, images_lr, label_strs, _ = data
            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            prob_str_lr = []
            for j in range(images_lr.shape[0]):
                parseq_input = self.parse_parseq_data(images_lr[j, :3, :, :])
                parseq_output = parseq(parseq_input)
                pred = parseq_output.softmax(-1)
                prob_str_lr.append(pred)
            prob_str_lr = torch.cat(prob_str_lr, dim=0)

            if not self.args.pre_training:
                label_vecs_final_new = None
                for j in range(val_batch_size):
                    data_diff = {"SR": prob_str_lr[j, :, :].unsqueeze(0)}
                    self.diffusion.feed_data(data_diff)
                    self.diffusion.test()
                    visuals = self.diffusion.get_current_visuals()
                    prior = visuals['SR']
                    if label_vecs_final_new is None:
                        label_vecs_final_new = prior
                    else:
                        label_vecs_final_new = torch.concat([label_vecs_final_new, prior], dim=0)
                label_vecs_final_new = label_vecs_final_new.to(self.device)
            else:
                prob_str_hr = []
                for j in range(images_hr.shape[0]):
                    parseq_input = self.parse_parseq_data(images_hr[j, :3, :, :])
                    parseq_output = parseq(parseq_input)
                    pred = parseq_output.softmax(-1)
                    prob_str_hr.append(pred)
                label_vecs_final_new = torch.cat(prob_str_hr, dim=0)
            
            images_sr, _ = model(images_lr, label_vecs_final_new)
            
            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True, debug=True)
                preds, _ = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                aster_output_sr = aster(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output, _ = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
            for j in range(val_batch_size):
                if str_filt(pred_str_sr[j], 'lower') == str_filt(label_strs[j], 'lower'):
                    n_correct += 1
                    
            sum_images += val_batch_size
            torch.cuda.empty_cache()
            print('Evaluation: [{}][{} / {}]'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          i + 1, len(test_loader)))
            
        time_end = time.time()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        acc = round(n_correct / sum_images, 4)
        fps = sum_images/(time_end - time_begin)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        current_acc_dict[data_name] = float(acc)
        result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'fps': fps}
        print(result)
        print('='*110)
