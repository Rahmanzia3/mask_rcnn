import os
import csv
import glob
import requests
import datetime
import telegram
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from telegram.ext import Updater
from argparse import ArgumentParser


def send_tele(data):
    try :
        updater = Updater('805281461:AAH09xnakEe8MxOBLQ7jWaiNolGxoZyxxrM', use_context=True )
        dp = updater.dispatcher
        token='805281461:AAH09xnakEe8MxOBLQ7jWaiNolGxoZyxxrM'
        bot = telegram.Bot(token=token)
        bot.send_message(chat_id='-496362146', text=data+ str(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))
    except:
        print('SENDING ERROR IN plot')

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as bar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        bar.update(CHUNK_SIZE)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination) 


if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("--backbone", default='mask_rcnn', help="Which yolo you want to use mask_rcnn")
    parser.add_argument("--project", required=True, help="Give project name")
    parser.add_argument("--batch_size", default = 3 , help="Give batch size")
    parser.add_argument("--steps", default =  250, help="Give batch step size")
    parser.add_argument("--epochs", default =  25, help="Give epoch size or save weights at")
    parser.add_argument("--data_source", required = True, help='Provide url or google Id or path(dir)')

    opt = parser.parse_args()



    '''
    python detectron_mask_rcnn_train.py --data_source /home/tericsoft/team_alpha/data/img_dirs/balloon_dataset/balloon --project baloon_test

    '''

    data_source_get = opt.data_source
    project_get = opt.project
    batch_size_get = opt.batch_size
    steps_get = opt.steps
    epochs_get = opt.epochs

    ephocs_and_all = ' --batch_size '+str(batch_size_get) +' '+ '--steps '+str(steps_get)+' '+ '--epochs '+ str(epochs_get)


    run = 'python detectron_mask_rcnn_train.py '+'--data_source '+ data_source_get+' '+'--project '+project_get+ephocs_and_all

    p = subprocess.Popen(run, shell=True, stdout=subprocess.PIPE)
   

    once_send = 'Project : ' +str(project_get) +' Backbone : '+ str(opt.backbone) +'  ,'
    try:
        send_tele('Training Has Been Started ')
        print('training has been started')
        send_tele(once_send)
    except:
        pass

    while True:
        out = p.stdout.readline()
        all_op = out.decode("utf-8")

        is_itr_comp = all_op.find('eta:')
        
        is_ending = all_op.find('Total training time:')
        if is_itr_comp != -1:
            send_tele(all_op)
            print(all_op)
        if is_ending != -1:
            print('Training has been stopped')
            send_tele('Training has been stopped ')
            break


# 17OCk2sCV1rngxL1_-MjpB0gypF85fIKU