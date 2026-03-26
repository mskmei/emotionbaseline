import torch, time, math
import numpy as np
from pathlib import Path
from sklearn import metrics
from tqdm import tqdm


def init_weight(model, method='xavier_uniform_'):
    if method == 'xavier_uniform_': fc = torch.nn.init.xavier_uniform_
    if method == 'xavier_normal_':  fc = torch.nn.init.xavier_normal_
    if method == 'orthogonal_':     fc = torch.nn.init.orthogonal_

    for name, param in model.named_parameters():
        if 'plm' not in name: # 跳过 plm 模型参数
            if param.requires_grad:
                if len(param.shape) > 1: fc(param) # 参数维度大于 1
                else: 
                    stdv = 1. / math.sqrt(param.shape[0])
                    torch.nn.init.uniform_(param, a=-stdv, b=stdv)

def print_trainable_parameters(args, model):
    params_all, params_train = 0, 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"): num_params = param.ds_numel
        params_all += num_params
        if param.requires_grad: params_train += num_params
    
    p_train, p_all = f"{round(params_train/1000000, 2)} M", f"{round(params_all/1000000, 2)} M"
    train_rate = round(100*params_train/params_all, 2)

    args.logger['process'].warning(f"train: {p_train} || all params: {p_all} || trainable: {train_rate} %")

class Processor():
    def __init__(self, args, model, dataset) -> None:
        self.args = args
        self.dataset = dataset
        self.model = model.to(args.train['device'])
        init_weight(self.model) # 初始化模型参数
        print_trainable_parameters(args, self.model) # 打印训练参数比重

        if self.dataset.loader: self.dataloader = self.dataset.loader
        else: self.dataloader = self.dataset.get_dataloader(self.args.train['batch_size'])
        self.log_step_rate = args.train['log_step_rate']
        self.global_step = 1
        self.log_step = int(len(self.dataloader['train']) / self.log_step_rate)
        self.model.get_optimizer() # 初始化优化器

        for k, v in vars(args).items():
            for kk, vv in v.items(): args.logger['params'].info(f"{k}.{kk}: {vv}")
        args.logger['params'].info(f"\n {'='*120} \n")

        display = ''
        for item in args.logger['display']: 
            if item in args.train: display += f"{item}: {args.train[item]}, "
            if item in args.model: display += f"{item}: {args.model[item]}, "
        args.logger['process'].warning(display)

    def train_desc(self, epoch, ttime=None):
        args, metrics = self.args, self.dataset.metrics.results
        epochs, model_name, data_name = args.train['epochs'], args.model['name'], self.dataset.name[-1]
        m = self.dataset.metrics.base
        m_tr, m_vl, m_te = round(metrics['train'][m], 3), round(metrics['valid'][m], 3), round(metrics['test'][m], 3)
        m_tr_loss = round(metrics['train']['loss'], 3)
        desc = f"eh {epoch}/{epochs} ({model_name}=>{data_name}: {str(m_tr)}/{str(m_vl)}/{str(m_te)}, loss: {str(m_tr_loss)}, time: {ttime})"
        self.tqdm_epochs.set_description(desc)
        if epoch>=0: self.tqdm_epochs.update()

    def train_stop(self, epoch=None):
        metric_valid = self.dataset.metrics.results['valid']
        early_threshold = epoch-metric_valid['epoch'] if 'epoch' in metric_valid else 0

        # 0. 达到阈值，停止训练
        if early_threshold >= self.args.train['early_stop']:
            return True
        
        # 1. 长期未更新了，增加评价次数
        if early_threshold: 
            self.log_step_rate = self.args.train['log_step_rate']+early_threshold*0.5
            self.log_step_rate = min(self.log_step_rate, 3.0)
        else: self.log_step_rate = self.args.train['log_step_rate']

    def train_batch(self, batch, bi=None):
        self.model.train() 
        if isinstance(batch, dict):     
            for key, val in batch.items(): batch[key] = val.to(self.args.train['device'])
        if isinstance(batch, list):
            for i, val in enumerate(batch): batch[i] = val.to(self.args.train['device'])
        outs = self.model.training_step(batch, bi)  
        
        outs["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.train['max_grad_norm'])
        self.model.optimizer.step()
        if self.model.scheduler is not None: self.model.scheduler.step() 
        self.model.optimizer.zero_grad()        

        self.global_step += 1
        if self.global_step % self.log_step == 0:
            if self.args.train['do_valid']: self._evaluate(stage='valid')
            if self.args.train['do_test']:
                if self.args.train.get('test_every_epoch', False):
                    self._evaluate(stage='test')
                elif self.model.valid_update:
                    self._evaluate(stage='test')

    def _train(self):
        epochs, e_start = self.args.train['epochs'], self.args.train['e_start'] if 'e_start' in self.args.train else 0
        self.tqdm_epochs = tqdm(total=epochs, position=0) # 进度条
        self.tqdm_epochs.update(e_start)
        self.train_desc(epoch=-1) # initialize process bar
        if self.args.model['epoch_before']: self.model.epoch_deal()
        for epoch in range(e_start, epochs):
            s_time = time.time()
            self.model.cur_epoch = epoch
            if self.args.model['epoch_every']: self.model.epoch_deal(epoch)
            
            torch.cuda.empty_cache()
            if self.args.train['show']: # 显示每个epoch的进度条
                for batch in tqdm(self.dataloader['train'], smoothing=0.05):
                    self.train_batch(batch, bi=-1)
            else: 
                for bi, batch in enumerate(self.dataloader['train']):
                    self.train_batch(batch, bi)
            
            if self.args.model['epoch_after']: self.model.epoch_deal(epoch)
            self.model.on_train_epoch_end()

            self.train_desc(epoch, round(time.time()-s_time, 1))
            if self.train_stop(epoch): break 
            
        self.tqdm_epochs.close()
        return self.dataset.metrics.results

    def _evaluate(self, stage='test'):
        # for bi, batch in enumerate(self.dataloader[stage]):
        #     with torch.no_grad():
        #         for key, val in batch.items(): batch[key] = val.to(self.args.train['device'])
        #         if stage == 'valid': self.model.validation_step(batch, bi)
        #         if stage == 'test': self.model.test_step(batch, bi)
            
        # if stage == 'valid': self.model.on_validation_end()
        # if stage == 'test': self.model.on_test_end()
        # return self.dataset.metrics
        self.model.eval()
        with torch.no_grad():
            if self.args.train['show']: # 显示每个epoch的进度条
                for batch in tqdm(self.dataloader[stage], smoothing=0.05):
                    if isinstance(batch, dict):
                        for key, val in batch.items(): batch[key] = val.to(self.args.train['device'])
                    if isinstance(batch, list):
                        for i, val in enumerate(batch): batch[i] = val.to(self.args.train['device'])
                    if stage == 'valid': self.model.validation_step(batch, -1)
                    if stage == 'test': self.model.test_step(batch, -1)
            else:
                for bi, batch in enumerate(self.dataloader[stage]):
                    if isinstance(batch, dict):
                        for key, val in batch.items(): batch[key] = val.to(self.args.train['device'])
                    if isinstance(batch, list):
                        for i, val in enumerate(batch): batch[i] = val.to(self.args.train['device'])
                    if stage == 'valid': self.model.validation_step(batch, bi)
                    if stage == 'test': self.model.test_step(batch, bi)
            
        if stage == 'valid':
            stage_outputs = list(self.model.validation_step_outputs)
            self._save_stage_report(stage, stage_outputs)
            self.model.on_validation_end()
        if stage == 'test':
            stage_outputs = list(self.model.test_step_outputs)
            self._save_stage_report(stage, stage_outputs)
            self.model.on_test_end()
        return self.dataset.metrics.results

    def _save_stage_report(self, stage, outputs):
        if not self.args.train.get('save_cls_report', False):
            self.args.logger['process'].warning(f"[{stage}] skip report: save_cls_report=False")
            return
        if outputs is None or len(outputs) == 0:
            self.args.logger['process'].warning(f"[{stage}] skip report: no outputs")
            return

        # Only for ERC-like classification outputs with preds/labels.
        if not isinstance(outputs[0], dict) or ('preds' not in outputs[0]) or ('labels' not in outputs[0]):
            self.args.logger['process'].warning(f"[{stage}] skip report: outputs missing preds/labels")
            return

        preds = np.concatenate([rec['preds'].detach().cpu().numpy() for rec in outputs])
        labels = np.concatenate([rec['labels'].detach().cpu().numpy() for rec in outputs])
        if len(labels) == 0:
            self.args.logger['process'].warning(f"[{stage}] skip report: empty labels")
            return

        label_map = self.dataset.tokenizer_['labels']['itol']
        report_style = self.args.train.get('report_style', '')

        if report_style == 'anjs':
            target_names = ['A', 'N', 'J', 'S']
            map_cfg = self.args.train.get('report_label_map', {})
            # Convert idx -> label name -> A/N/J/S
            mapped = []
            for g, p in zip(labels.tolist(), preds.tolist()):
                g_name = str(label_map[int(g)])
                p_name = str(label_map[int(p)])
                g2 = map_cfg.get(g_name)
                p2 = map_cfg.get(p_name)
                if g2 is None or p2 is None:
                    continue
                mapped.append((g2, p2))

            if not mapped:
                self.args.logger['process'].warning(f"[{stage}] skip report: no labels matched report_label_map")
                return

            anjs_to_idx = {k: i for i, k in enumerate(target_names)}
            labels_eval = np.array([anjs_to_idx[g] for g, _ in mapped], dtype=np.int64)
            preds_eval = np.array([anjs_to_idx[p] for _, p in mapped], dtype=np.int64)
            labels_idx = list(range(len(target_names)))
        else:
            labels_idx = sorted([int(k) for k in label_map.keys()])
            target_names = [str(label_map[i]) for i in labels_idx]
            labels_eval = labels.astype(np.int64)
            preds_eval = preds.astype(np.int64)

        report = metrics.classification_report(
            labels_eval,
            preds_eval,
            labels=labels_idx,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
        cm = metrics.confusion_matrix(labels_eval, preds_eval, labels=labels_idx)
        support = cm.sum(axis=1).tolist()
        gold_counts = np.bincount(labels_eval, minlength=len(labels_idx)).tolist()
        pred_counts = np.bincount(preds_eval, minlength=len(labels_idx)).tolist()
        acc = float((labels_eval == preds_eval).mean())

        save_dir = Path(self.args.train.get('report_dir', self.args.file['save_dir']))
        save_dir.mkdir(parents=True, exist_ok=True)
        ep = int(self.model.cur_epoch) + 1
        prefix = f"{stage}_epoch{ep:03d}"

        np.save(save_dir / f"{prefix}_confusion_matrix.npy", cm)
        with open(save_dir / f"{prefix}_confusion_matrix.txt", 'w', encoding='utf-8') as f:
            f.write('labels: ' + ','.join(target_names) + '\n')
            for row in cm:
                f.write(' '.join(str(int(x)) for x in row) + '\n')
        with open(save_dir / f"{prefix}_classification_report.txt", 'w', encoding='utf-8') as f:
            f.write(f"epoch={ep}\n")
            f.write(f"stage={stage}\n")
            f.write(f"accuracy={acc:.6f}\n")
            if report_style == 'anjs':
                f.write(f"support(A,N,J,S)={support} total={sum(support)}\n")
                f.write(f"gold_counts(A,N,J,S)={gold_counts}\n")
                f.write(f"pred_counts(A,N,J,S)={pred_counts}\n\n")
            else:
                f.write(f"support={support}\n")
                f.write(f"gold_counts={gold_counts}\n")
                f.write(f"pred_counts={pred_counts}\n\n")
            f.write(report)

        self.args.logger['process'].warning(
            f"[{stage}] report saved: {save_dir / (prefix + '_classification_report.txt')}"
        )