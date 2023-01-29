from utility.cross_validation import split_5_folds
from configx.configx import ConfigX

if __name__ == "__main__":
    configx = ConfigX()
    configx.k_fold_num = 5 
    configx.rating_path = "../data/ft_ratings.txt"
    configx.rating_cv_path = "../data/cv/"
    
    split_5_folds(configx)