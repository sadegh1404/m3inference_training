from m3inference.train import M3Trainer
import argparse


parser = argparse.ArgumentParser(description='M3 training')

parser.add_argument('--train_data_path',type=str)
parser.add_argument('--save_model_path',type=str,default='./m3model.pth',required=False)
args = parser.parse_args()

if __name__ == "__main__":
    trainer = M3Trainer(args.train_data_path)
    trainer.train()
    trainer.save(args.save_model_path)