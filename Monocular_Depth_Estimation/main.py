from Solver import Solver
import argparse

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help="Indicate if the training needs to be resumed from the last saved checkpoint")
    opt = parser.parse_args()
    return opt

if __name__=='__main__':
    opt = get_params()
    solver = Solver(opt)
    solver.train()