import argparse
import torch
import numpy as np
import random
import pytorch_lightning as pl
import pickle
from tqdm import tqdm
import os
#from torchmetrics import CatMetric, MetricCollection, MetricTracker

from transformers import AutoTokenizer
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForMaskedLM, AutoModelForCausalLM, default_data_collator
from pytorch_lightning.callbacks import ModelCheckpoint


from plt_model import Wrapper
from dataset import WikiText2Dataset, MyDataLoader, prepare_data, preprocess_datasets

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback

def plt_model_load(model, checkpoint):
    state_dict = torch.load(checkpoint)['state_dict']
    model.load_state_dict(state_dict)
    return model


def load_model(load_path, plt_model):
    if load_path is not None:
        if load_path.endswith(".ckpt"):
            checkpoint = load_path
        else:
            if load_path.endswith("/"):
                checkpoint = load_path + "best.ckpt"
            else:
                raise ValueError(
                    "if it is a directory, if must end with /; if it is a file, it must end with .ckpt"
                )
        plt_model = plt_model_load(plt_model, checkpoint)
        print(f"Loaded model from {checkpoint}")
    return plt_model


def main():
    arguments = {
        ('-tag', '--model_tag'): {
            'type': str,
            'default': 'facebook/opt-125m',
            'help': 'Model tag for the model to use.',
        },

        # checkpoint
        ('-load', '--load-name'): {
            'type': str,
            'default': None,
            'help': 'Name of the saved model to restore.'
        },
        ('-save', '--save-name'): {
            'type': str,
            'default': None,
            'help': 'Name of the saved model to save.'
        },
        ('-saveperplexities', '--saveperplexities'): {
            'type': str,
            'default': None,
            'help': 'Save individual perplexituy scores',
        },
        # debug control
        ('-eval', '--eval_only'): {
            'action': 'store_true',
            'help': 'Only run eval on the test set',
        },
        ('-eval_gen', '--evalgen_only'): {
            'action': 'store_true',
            'help': 'Only run eval on the test set',
        },
        # common training args
        ('-opt', '--optimizer'): {
            'type': str,
            'default': 'adamw',
            'help': 'Pick an optimizer.',
        },
        ('-lr', '--learning-rate'): {
            'type': float,
            'default': 2e-5,
            'help': 'Initial learning rate.',
        },
        ('-m', '--max-epochs'): {
            'type': int,
            #'default': 30,#100,
            'default': 5,#100,
            'help': 'Maximum number of epochs for training.',
        },
        ('-st', '--saveto'): {
            'type': str,
            'default': None,
            'help': 'Save the evaluation result to what location',
        },
        ('-b', '--batch-size'): {
            'type': int,
            'default': 128,
            'help': 'Batch size for training and evaluation.',
        },

        # debug control
        ('-d', '--debug'): {
            'action': 'store_true',
            'help': 'Verbose debug',
        },
        ('-seed', '--seed'): {
            'type': int,
            'default': 0,
            'help': 'Number of steps for model optimisation',
        },

        # cpu gpu setup for lightning
        ('-w', '--num_workers'): {
            # multiprocessing fail with too many works
            'type': int,
            'default': 1,
            'help': 'Number of CPU workers.',
        },
        ('-n', '--num_devices'): {
            'type': int,
            'default': 1,
            'help': 'Number of GPU devices.',
        },
        ('-a', '--accelerator'): {
            'type': str,
            'default': None,
            'help': 'Accelerator style.',
        },
        ('-s', '--strategy'): {
            'type': str,
            'default': 'ddp',
            #'default': 'ddp_find_unused_parameters_false',
            'help': 'Strategy style.',
        },

        ('-p', '--pretrained'): {
            'action': 'store_true',
            'help': 'Load a pretrained network from Huggingface',
        },

        ('-version_name', '--version_name'): {
            'type': str,
            'default': None,
            'help': 'Version name.',
        },

        # custom dataset generation and loading
        ('-gen', '--generate'): {
            'type': str,
            'default': None,
            'help': 'The file name to store the generated dataset.',
        },
        ('-load-gen', '--load-generate'): {
            'type': str,
            'default': None,
            'help': 'The file name to load the generated dataset.',
        },
        ('-gen_percent', '--generate_percentage'): {
            'type': float,
            'default': 0.0,
            'help': 'How much to mix from the trained model',
        },
    }


    p = argparse.ArgumentParser(
        description='GPT-N')
    for k, v in arguments.items():
        p.add_argument(*k, **v)
    a = p.parse_args()

    if (a.saveto is not None) and os.path.exists(a.saveto):
        exit()

    # seeding
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    model_tag = a.model_tag
    #model_tag = "lnair/opt-1.3b-wikitext2"
    # model_tag = "gpt2"
    if a.pretrained:
        model = AutoModelForCausalLM.from_pretrained(model_tag, cache_dir='./data_cache_dir',)
    else:
        config = AutoConfig.from_pretrained(model_tag)
        model = AutoModelForCausalLM.from_config(config=config)
    print(f"Loaded model from HuggingFace")

    tokenizer = AutoTokenizer.from_pretrained(model_tag,
                                              cache_dir='./model_cache_dir',
                                              return_dict=True)
    print(f"Loaded tokenizer from HuggingFace")

    raw_dataset = prepare_data()
    dataset = preprocess_datasets(
        raw_dataset, tokenizer)

    if a.load_generate is not None:
        with open(a.load_generate, 'rb') as f:
            train_dataset = pickle.load(f)
        print(f"Loaded dataset from {a.load_generate}")
    else:
        train_dataset = WikiText2Dataset(dataset=dataset, partition='train', tokenizer=tokenizer)
        #evens = list(range(0, 1000))
        #train_dataset = torch.utils.data.Subset(train_dataset, evens)

    val_dataset = WikiText2Dataset(dataset=dataset, partition='validation', tokenizer=tokenizer)
    test_dataset = WikiText2Dataset(dataset=dataset, partition='test', tokenizer=tokenizer)

    #evens = list(range(0, 1000))
    #train_dataset = torch.utils.data.Subset(train_dataset, evens)

    if a.evalgen_only:
        test_dataset = train_dataset

    data_loader = MyDataLoader(
        'wikitext2',
        a.num_workers,
        train_dataset, 
        val_dataset, 
        test_dataset, 
        batch_size=a.batch_size)
    
    #print(f"Loaded dataset from HuggingFace")

    checkpoint_callback=ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="best",
        dirpath=a.save_name,
        save_last=True,
    )

    plt_model = Wrapper(
        model,
        learning_rate=a.learning_rate,
        epochs=a.max_epochs)
    
    if a.version_name is not None:
        logger = TensorBoardLogger('./', version=a.version_name)
    else:
        logger = TensorBoardLogger('./')

    plt_model=load_model(plt_model=plt_model, load_path=a.load_name)
    #testm = MetricTracker()#MetricTracker(MetricCollection([CatMetric()]))


    trainer=pl.Trainer(
        max_epochs=a.max_epochs,
        devices=a.num_devices,
        accelerator=a.accelerator,
        strategy=a.strategy,
        fast_dev_run=a.debug,
        callbacks=[checkpoint_callback],#, testm],
        logger=logger,
    )
    
    plt_model = plt_model.cuda()
    if not a.eval_only:
        trainer.fit(
            plt_model, 
            train_dataloaders=data_loader.train_dataloader,
            val_dataloaders=data_loader.val_dataloader)
    
    if a.saveperplexities:
        plt_model.tosave=True

    res = trainer.test(plt_model, dataloaders=data_loader.test_dataloader)

    if a.saveperplexities is not None:
        with open(a.saveperplexities, "wb") as f:
            pickle.dump(plt_model.saved, f)

    if a.saveto is not None:
        with open(a.saveto, "wb") as f:
            pickle.dump(res, f)
    
    if a.generate is not None:
        with torch.no_grad():
            limit = len(data_loader.train_dataset) * a.generate_percentage / a.batch_size
            i = 0
            model = plt_model.model.eval()
            batches = []
            for batch in tqdm(data_loader.train_dataloader):
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                labels = batch["labels"].cuda()
                plt_model = plt_model.cuda()
                model = model.cuda()

                batch_size, _ = input_ids.shape
                if i >= limit:
                    #outputs = model(
                    #    input_ids=input_ids, 
                    #    attention_mask=attention_mask, 
                    #    labels=labels)
                    #preds = outputs.logits.argmax(-1)

                    outputs = model.generate(input_ids, num_beams=5, max_new_tokens=64, min_new_tokens=64, repetition_penalty=3.0)
                    preds = outputs[:, 64:]

                    #print(f"IInput {len(input_ids[0])}:", tokenizer.decode(input_ids[0]))
                    #print(f"Labels {len(labels[0])}:", tokenizer.decode(labels[0]))
                    #print(f"Output {len(preds[0])}:", tokenizer.decode(preds[0]))
                    #print(attention_mask[0])
                    #print(f"-----")

                    batch["input_ids"] = preds.cpu().detach()
                    batch["labels"]    = preds.cpu().detach()
                    batch["attention_mask"] = attention_mask.cpu().detach()

                #print("Input:", tokenizer.decode(input_ids[0]))
                #print(f"Output {len(labels[0])}:", tokenizer.decode(labels[0]))
                my_batch = []
                for j in range(batch_size):
                    my_batch.append({"input_ids": batch["input_ids"][j, :], "attention_mask": batch["attention_mask"][j, :], "labels": batch["labels"][j, :]})
                batches += my_batch
                i += 1
            file_name = f"{a.generate}.pkl"
            with open(file_name, "wb") as f:
                pickle.dump(batches, f)
            print(f"{file_name} generated and saved!")


if __name__ == '__main__':
    main()
