import torch
import wget
from pathlib import Path
import zipfile
import pandas as pd

import time
import datetime
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random

from sklearn.metrics import matthews_corrcoef
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer

def get_device():
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

def download_data():
    print('Downloading dataset...')
    # The URL for the dataset zip file.
    url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
    # Download the file (if we haven't already)
    data_folder = "../data"
    data_folder = Path(data_folder).resolve()
    data_folder.mkdir(parents=True, exist_ok=True)
    zip_file_path = data_folder / "cola_public_1.1.zip"
    wget.download(url, zip_file_path.as_posix())

    output_dir = data_folder/"cola"
    output_dir.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(zip_file_path.as_posix(), 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    return output_dir


def set_seed(seed_val=1024):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def get_traindata(data_dir):
    # read the train data
    # Load the dataset into a pandas dataframe.
    df = pd.read_csv(data_dir/"cola_public/raw/in_domain_train.tsv",
                     delimiter='\t', header=None,
                     names=['sentence_source', 'label', 'label_notes', 'sentence'])

    sentences = df.sentence.values
    labels = df.label.values
    return sentences, labels


def data_preprocess(tokenizer, sentences, labels, max_len=64):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 64,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset


def get_dataloader(dataset, batch_size=32, valid=True):
    # Create a 90-10 train-validation split.
    # Calculate the number of samples to include in each set.
    if valid:
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        train_dataset = dataset

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    if not valid: return train_dataloader, ''
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

    return train_dataloader, validation_dataloader

def preprocess(sentences, labels, tokenizer, valid=False):
    train_dataset = data_preprocess(tokenizer, sentences, labels)
    train_dataloader, valid_dataloader = get_dataloader(train_dataset, valid=valid)
    return train_dataloader, valid_dataloader


def bert_model():
    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    # Tell pytorch to run this model on the GPU.
    model.cuda()

    optimizer = AdamW(model.parameters(),
                                        lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                                        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                                      )

    return model, optimizer

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(train_dataloader, model, optimizer, scheduler, device, epochs=4):
    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))


def validate(validation_dataloader, device, model):
    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    predictions , true_labels = [], []
    # Evaluate data for one epoch
    for batch in validation_dataloader:

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            (loss, logits) = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)
        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)


    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
    return predictions, true_labels

def modelling(train_dataloader, valid_dataloader, device):
    model, optimizer = bert_model()

    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
    # do training
    train(train_dataloader, model, optimizer, scheduler, device, epochs)
    if valid_dataloader:
        validate(valid_dataloader, device, model)
    return model

def inference(data_dir, tokenizer, model, device):
    # Load the dataset into a pandas dataframe.
    df = pd.read_csv(data_dir/"cola_public/raw/out_of_domain_dev.tsv",
                     delimiter='\t', header=None,
                     names=['sentence_source', 'label', 'label_notes', 'sentence'])

    # Create sentence and label lists
    sentences = df.sentence.values
    labels = df.label.values
    test_dataloader, _ = preprocess(sentences, labels, tokenizer, valid=False)
    predictions, true_labels = validate(test_dataloader, model, device)

    matthews_set = []

    # Evaluate each test batch using Matthew's correlation coefficient
    print('Calculating Matthews Corr. Coef. for each batch...')

    # For each input batch...
    for i in range(len(true_labels)):
        # The predictions for this batch are a 2-column ndarray (one column for "0"
        # and one column for "1"). Pick the label with the highest value and turn this
        # in to a list of 0s and 1s.
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
        # Calculate and store the coef for this batch.
        matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
        matthews_set.append(matthews)
    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

    print('Total MCC: %.3f' % mcc)


def main():
    data_dir = download_data()
    device = get_device()
    sentences, labels = get_traindata(data_dir)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)
    valid = True
    train_dataloader, valid_dataloader = preprocess(sentences, labels,
                                                    tokenizer, valid=valid)
    if valid:
        model = modelling(train_dataloader, valid_dataloader, device=device)
    else:
        model = modelling(train_dataloader, valid_dataloader=None, device=device)

    # inference data
    inference(data_dir, tokenizer, model, device)
