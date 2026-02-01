from pathlib import Path
import numpy as np
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader
import pandas as pd
import warnings
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold


from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from .dataset import MyDataset
import os 

tqdm.pandas()




class MyDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.input_dir = Path(__file__).parents[1].joinpath("input")
        self.output_dir = Path(__file__).parents[1].joinpath("output")

    def prepare_data(self):
        pass

    def prepare_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        df[['sample_id_prefix', 'sample_id_suffix']] = df.sample_id.str.split('__', expand=True)
        (df.sample_id_suffix == df.target_name).all()

        cols = ['sample_id_prefix', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']

        # 1つのデータについて、5行のターゲットがあるので、グルーピングして1つにまとめる
        agg_df = df.groupby(cols).apply(lambda df: df.set_index('target_name').target)
        
        agg_df.reset_index(inplace=True)
        agg_df.columns.name = None

        agg_df['image'] = agg_df.image_path.progress_apply(
            lambda path: Image.open(self.input_dir / "csiro-biomass" / path).convert('RGB')
        )
        agg_df['image_size'] = agg_df.image.apply(lambda x: x.size)
        
        print(agg_df['image_size'].value_counts())
        print(
            np.isclose(
                agg_df[['Dry_Green_g', 'Dry_Clover_g']].sum(axis=1),
                agg_df['GDM_g'],
                atol=1e-04
            ).mean()
        )
        print(
            np.isclose(
                agg_df[['GDM_g', 'Dry_Dead_g']].sum(axis=1),
                agg_df['Dry_Total_g'],
                atol=1e-04
            ).mean()
        )

        return agg_df

    def assign_folds(self, df, NFOLD) -> tuple[pd.DataFrame, pd.DataFrame]:
        kfold = KFold(n_splits=NFOLD, shuffle=True, random_state=42)

        # 検証が重複しないようにするためのfold列を追加し、番号を割り当てる
        df['fold'] = None
        for i, (trn_idx, val_idx) in enumerate(kfold.split(df.index)):
            df.loc[val_idx, 'fold'] = i
        return df

    def setup(self, stage=None):
        df = pd.read_csv(self.input_dir.joinpath("csiro-biomass","train.csv"))
        df = self.prepare_samples(df)

        # 既存 fold CSV を読み込む
        folds_df = pd.read_csv(
            self.input_dir / "csiro-biomass" / "folds_opt_ver5.csv"
        )

        df = df.merge(
            folds_df[['sample_id_prefix', 'fold']],
            on='sample_id_prefix',
            how='left'
        )
        
        if df['fold'].isna().any():
            missing = df[df['fold'].isna()]['sample_id_prefix'].unique()
            raise ValueError(
                f"fold が割り当てられていない sample_id_prefix があります: {missing[:5]}"
            )

        self.df = df

        train_df = self.df[self.df['fold'] != 0].reset_index(drop=True)
        val_df   = self.df[self.df['fold'] == 0].reset_index(drop=True)
        
        self.df = df

        train_df = self.df[self.df['fold'] != 0].reset_index(drop=True)
        val_df   = self.df[self.df['fold'] == 0].reset_index(drop=True)
        
        self.train_dataset = MyDataset(data=train_df, cfg=self.cfg, mode="train")
        self.val_dataset = MyDataset(data=val_df, cfg=self.cfg, mode="val")
        print(f"train: {len(self.train_dataset)}, val: {len(self.val_dataset)}")

    def train_dataloader(self):    
        return DataLoader(self.train_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=True, drop_last=True, num_workers=self.cfg.data.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers, pin_memory=True)
