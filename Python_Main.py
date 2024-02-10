import os
import sys
import pickle
import random
import copy
import numpy as np
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup, RobertaModel

# Importing custom modules
from utils import Config, Logger, make_log_dir
from modeling import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification_SPV,
    AutoModelForSequenceClassification_MIP,
    AutoModelForSequenceClassification_SPV_MIP,
)
from run_classifier_dataset_utils import processors, output_modes, compute_metrics
from data_loader import load_train_data, load_train_data_kf, load_test_data

# Constant file names
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
ARGS_NAME = "training_args.bin"

# def calculate_metrics(preds, labels):
#     precision = precision_score(labels, preds, average='weighted')
#     recall = recall_score(labels, preds, average='weighted')
#     f1 = f1_score(labels, preds, average='weighted')
#     return precision, recall, f1

def add_cos(list_, cos):
    """
    Add cosine similarity values to a list.

    Args:
        list_: List to which cosine similarity values are added.
        cos: Cosine similarity values.

    Returns:
        Updated list with added cosine similarity values.
    """
    if len(list_) == 0:
        list_.append(cos.detach().cpu().numpy())
    else:
        list_[0] = np.append(list_[0], cos.detach().cpu().numpy())
    return list_

def calcu_cos(mip_cos, bmip_cos):
    """
    Calculate and print mean cosine similarity values.

    Args:
        mip_cos: List containing cosine similarity values for MIP.
        bmip_cos: List containing cosine similarity values for BMIP.
    """
    if len(mip_cos) > 0 and len(bmip_cos) > 0:
        m_mean = np.mean(mip_cos[0])
        bm_mean = np.mean(bmip_cos[0])
        print(f'mip cos mean: {m_mean}; bmip cos mean: {bm_mean}')
        print(f'{len(mip_cos[0])}, {len(bmip_cos[0])}')
        print(mip_cos[0], bmip_cos[0])
    else:
        print("mip_cos or bmip_cos is empty!")

def save_preds_npy(args, preds, labels):
    """
    Save predictions and labels as numpy files.

    Args:
        args (Namespace): Training arguments.
        preds: Model predictions.
        labels: Ground truth labels.
    """
    path = os.path.join(args.log_dir, 'preds.npy')
    np.save(path, preds)
    path = os.path.join(args.log_dir, 'labels.npy')
    np.save(path, labels)

def load_pretrained_model(args):
    """
    Load a pretrained model based on the provided arguments.

    Args:
        args (Namespace): Training arguments.

    Returns:
        Loaded model.
    """
    # Pretrained Model
    bert = AutoModel.from_pretrained(args.bert_model)
    config = bert.config
    config.type_vocab_size = 4
    if "albert" in args.bert_model:
        bert.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_size
        )
    else:
        bert.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
    bert._init_weights(bert.embeddings.token_type_embeddings)

    # Additional Layers
    if args.model_type in ["BERT_BASE"]:
        model = AutoModelForSequenceClassification(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "BERT_SEQ":
        model = AutoModelForTokenClassification(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT_SPV":
        model = AutoModelForSequenceClassification_SPV(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT_MIP":
        model = AutoModelForSequenceClassification_MIP(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT":
        _model_path = '../../WSCMM/models/senseCL/checkpoint/checkpoint-1200/'
        basic_encoder = get_model(_model_path)
        model = AutoModelForSequenceClassification_SPV_MIP(
            args=args, Model=bert, basic_encoder=basic_encoder, config=config, num_labels=args.num_labels
        )

    model.to(args.device)
    if args.n_gpu > 1 and not args.no_cuda:
        model = torch.nn.DataParallel(model)
    return model

def load_json(path):
    """
    Load JSON data from a file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Loaded JSON data.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_config(path):
    """
    Read and return configuration data from a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Configuration data.
    """
    config_path = os.path.join(path, 'config.json')
    config = load_json(config_path)
    print(config)
    return config

def get_model(path):
    """
    Load and return a model based on the provided path.

    Args:
        path (str): Path to the model checkpoint.

    Returns:
        Loaded model.
    """
    model = None
    config = read_config(path)
    try:
        config = read_config(path)
    except:
        print('===========Fail to load config.json!============')
    if config['_name_or_path'] == 'roberta-base':
        model = RobertaModel.from_pretrained(path)

    if model == None:
        print('===========Fail to load Model!============')
    
    return model

def save_model(args, model, tokenizer):
    """
    Save the trained model, configuration, and tokenizer.

    Args:
        args (Namespace): Training arguments.
        model: Trained model.
        tokenizer: Tokenizer used for training.
    """
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.log_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.log_dir)

    # Good practice: save your training arguments together with the trained model
    output_args_file = os.path.join(args.log_dir, ARGS_NAME)
    torch.save(args, output_args_file)

def load_trained_model(args, model, tokenizer):
    """
    Load a trained model and return it.

    Args:
        args (Namespace): Training arguments.
        model: Model to load weights into.
        tokenizer: Tokenizer used for training.

    Returns:
        Loaded model.
    """
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)

    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(output_model_file))
    else:
        model.load_state_dict(torch.load(output_model_file))

    return model

def loss_plot(train_loss, val_loss):
    """
    Plot the training and validation loss.

    Args:
        train_loss: List of training loss values.
        val_loss: List of validation loss values.
    """
    plt.plot(train_loss, label='Train loss')
    plt.plot(val_loss, label='Dev loss')

    plt.title('Change in Loss Per 500 step')
    plt.xlabel('*500 Steps')
    plt.ylabel('Loss')

    plt.legend()
    plot_path = 'saves/train_loss.png'
    plt.savefig(plot_path)
    
def Train(args, logger, model, train_dataloader, processor, task_name, label_list, tokenizer, output_mode, k=None):
    tr_loss = 0
    num_train_optimization_steps = len(train_dataloader) * args.num_train_epoch

    # Prepare optimizer, scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.lr_schedule != False and args.lr_schedule.lower() != "none":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_epoch * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
        )

    logger.info("***** Running training *****")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num steps = {num_train_optimization_steps}")

    # Run training
    model.train()
    max_val_f1 = -1
    max_result = {}
    train_loss = []
    val_loss = []
    log_loss = 0
    total_step = 0

    for epoch in trange(int(args.num_train_epoch), desc="Epoch"):
        tr_loss = 0

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # move batch data to gpu
            batch = tuple(t.to(args.device) for t in batch)

            if args.model_type in ["MELBERT_MIP", "MELBERT"]:
                # Unpack the batch for MELBERT models
                (
                    input_ids,
                    input_mask,
                    segment_ids,
                    label_ids,
                    input_ids_2,
                    input_mask_2,
                    segment_ids_2,
                    _input_ids,
                    _input_mask,
                    _segment_ids,
                ) = batch
            else:
                input_ids, input_mask, segment_ids, label_ids = batch

            # Forward pass
            if args.model_type in ["BERT_SEQ", "BERT_BASE", "MELBERT_SPV"]:
                # Transformer Encoding for BERT-based Models
                logits = model(
                    input_ids,
                    target_mask=(segment_ids == 1),
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
            
            elif args.model_type in ["MELBERT_MIP", "MELBERT"]:
                # Transformer Encoding for MELBERT Models
                # print("++++++++++++++++++++++++++++++++++++++++++++++++++")
                logits = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    basic_ids=_input_ids,
                    basic_mask=(_segment_ids == 1),
                    basic_attention=_input_mask,
                    basic_token_type_ids=_segment_ids,
                )

            # Compute loss
            loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
            loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))

            # Average loss if on multi-gpu.
            if args.n_gpu > 1:
                loss = loss.mean()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.lr_schedule != False and args.lr_schedule.lower() != "none":
                scheduler.step()

            optimizer.zero_grad()

            tr_loss += loss.item()
            ##############################
            log_loss += loss.item()

            # Log and save model at specific steps
            total_step += 1
            if total_step % 2000 == 0:
                logger.info(f'\nTrain loss for 2000 steps: {log_loss}')
                log_loss = 0

        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[epoch {epoch+1}] ,lr: {cur_lr} ,tr_loss: {tr_loss}")

        # Evaluate
        if args.do_eval:
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            # result = evaluation(args, logger, model, eval_dataloader, all_guids, task_name, tokenizer, return_loss=True, out_cos=True)
            result = evaluation(args, logger, model, eval_dataloader, all_guids, task_name)
            # Save the model during evaluation
            # if args.task_name == "vua":
            #     save_model(args, model, tokenizer)

            # Update best result
            # if isinstance(result, dict) and "f1" in result:
            
            if result["f1"] > max_val_f1:
                max_val_f1 = result["f1"]
                max_result = result
                if args.task_name == "trofi":
                    # =======================
                    # Save Model (SPV)
                    save_model(args, model, tokenizer)
            if args.task_name == "vua":
                # =======================
                # Save Model (MIP)
                save_model(args, model, tokenizer)

    logger.info("-----Best Result-----")
    for key in sorted(max_result.keys()):
        logger.info(f"  {key} = {str(max_result[key])}")
        # print("From Train Function: ")
        # print(f" {key} = {str(result[key])}")
    loss_plot(train_loss, val_loss)

    return model, max_result

# def evaluation(args, logger, model, eval_dataloader, all_guids, task_name, tokenizer, return_preds=False, return_loss=False, out_cos=False):
def evaluation(args, logger, model, eval_dataloader, all_guids, task_name, return_preds=False, return_loss=False):
    """
    Evaluate the model on the given dataloader.
    """

    model.eval()
    mip_cos = []
    bmip_cos = []
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    pred_guids = []
    out_label_ids = None

    for eval_batch in tqdm(eval_dataloader, desc="Evaluating"):
        eval_batch = tuple(t.to(args.device) for t in eval_batch)

        if args.model_type in ["MELBERT_MIP", "MELBERT"]:
            (
                input_ids,
                input_mask,
                segment_ids,
                label_ids,
                idx,
                input_ids_2,
                input_mask_2,
                segment_ids_2,
                _input_ids,
                _input_mask,
                _segment_ids,
            ) = eval_batch
        else:
            input_ids, input_mask, segment_ids, label_ids, idx = eval_batch

        with torch.no_grad():
            # compute loss values
            if args.model_type in ["BERT_BASE", "BERT_SEQ", "MELBERT_SPV"]:
                logits = model(
                    input_ids,
                    target_mask=(segment_ids == 1),
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    pred_guids.append([all_guids[i] for i in idx])
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                    pred_guids[0].extend([all_guids[i] for i in idx])
                    out_label_ids = np.append(
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                    )
            elif args.model_type in ["MELBERT_MIP", "MELBERT"]:
                if args.out_cos: 
                    logits, cos0, cos1 = model(
                        input_ids,
                        input_ids_2,
                        target_mask=(segment_ids == 1),
                        target_mask_2=segment_ids_2,
                        attention_mask_2=input_mask_2,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        basic_ids=_input_ids,
                        basic_mask=(_segment_ids==1),
                        basic_attention=_input_mask,
                        basic_token_type_ids=_segment_ids,
                    )
                    mip_cos = add_cos(mip_cos, cos0)
                    bmip_cos = add_cos(bmip_cos, cos1)
                else:
                    logits = model(
                        input_ids,
                        input_ids_2,
                        target_mask=(segment_ids == 1),
                        target_mask_2=segment_ids_2,
                        attention_mask_2=input_mask_2,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        basic_ids=_input_ids,
                        basic_mask=(_segment_ids==1),
                        basic_attention=_input_mask,
                        basic_token_type_ids=_segment_ids,
                    )

                loss_fct = nn.NLLLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    pred_guids.append([all_guids[i] for i in idx])
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                    pred_guids[0].extend([all_guids[i] for i in idx])
                    out_label_ids = np.append(
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                    )

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)
    if return_loss:
        return eval_loss

    save_preds_npy(args, preds, out_label_ids)

    # compute metrics
    result = compute_metrics(preds, out_label_ids)

    for key in sorted(result.keys()):
        logger.info(f"  {key} = {str(result[key])}")
        # print("From Evaluation Function: ")
        # print(f" {key} = {str(result[key])}")
    logger.info('out cos similarity!')
    calcu_cos(mip_cos, bmip_cos)

    # Save the model after evaluation in the "Eval_Model" directory
    # save_dir = os.path.join(args.output_dir, "Eval_Model")
    # os.makedirs(save_dir, exist_ok=True)
    # save_model(args, model, tokenizer)

    if return_preds:
        return preds
    return result

def main():
    config = Config(main_conf_path="./")

    # Apply system arguments if exist
    argv = sys.argv[1:]
    if len(argv) > 0:
        cmd_arg = OrderedDict()
        argvs = " ".join(sys.argv[1:]).split(" ")
        for i in range(0, len(argvs), 2):
            arg_name, arg_value = argvs[i], argvs[i + 1]
            arg_name = arg_name.strip("-")
            cmd_arg[arg_name] = arg_value
        config.update_params(cmd_arg)

    args = config
    args.num_train_epoch = 20 ## Set the number of training epochs to 10
    print(args.__dict__)

    # Logger
    if "saves" in args.bert_model:
        log_dir = args.bert_model
        logger = Logger(log_dir)
        config = Config(main_conf_path=log_dir)
        old_args = copy.deepcopy(args)
        args.__dict__.update(config.__dict__)

        args.bert_model = old_args.bert_model
        args.do_train = old_args.do_train
        args.data_dir = old_args.data_dir
        args.task_name = old_args.task_name

        # Apply system arguments if exist
        argv = sys.argv[1:]
        if len(argv) > 0:
            cmd_arg = OrderedDict()
            argvs = " ".join(sys.argv[1:]).split(" ")
            for i in range(0, len(argvs), 2):
                arg_name, arg_value = argvs[i], argvs[i + 1]
                arg_name = arg_name.strip("-")
                cmd_arg[arg_name] = arg_value
            config.update_params(cmd_arg)
    else:
        if not os.path.exists("saves"):
            os.mkdir("saves")
        log_dir = make_log_dir(os.path.join("saves", args.bert_model))
        logger = Logger(log_dir)
        config.save(log_dir)
    args.log_dir = log_dir

    # Set CUDA devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logger.info("Device: {} n_gpu: {}".format(device, args.n_gpu))

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Get dataset and processor
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    args.num_labels = len(label_list)

    # Build tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = load_pretrained_model(args)
    # model = AutoModel.from_pretrained(args.bert_model, num_labels=args.num_labels)
    print("SKIPING this Block....")
    # Load Data
    train_dataloader = load_train_data(args, logger, processor, task_name, label_list, tokenizer, output_mode)
    _, test_dataloader = load_test_data(args, logger, processor, task_name, label_list, tokenizer, output_mode)
    
    # # print("=======================Training Data============================================")
    # # for batch in train_dataloader:
    # #     print(batch)
    # #     break  # Print only the first batch for brevity

    # # print("\n ====================TEST DATA================================================")
    # # for batch in test_dataloader:
    # #     print(batch)
    # #     break  # Print only the first batch for brevity

    # # model = load_pretrained_model(args)
    # # Run training
    trained_model, best_result = Train(
        args,
        logger,
        model,
        train_dataloader,
        processor,
        task_name,
        label_list,
        tokenizer,
        output_mode
    )
    print(f"Best Result from Main function: {best_result}")

if __name__ == "__main__":
    main()
