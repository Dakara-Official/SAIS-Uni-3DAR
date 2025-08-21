# 以下仿照源代码crystal_grid_dataset.py构建的molecule_energy_grid_dataset.py
import torch
from .base_grid_dataset import BaseGridDataset
from . import molecule_energy_data_utils
from .atom_dictionary import RawAtomFeature, AtomGridFeature
import numpy as np

class MoleculeEnergyGridDataset(BaseGridDataset):
    def __init__(
        self,
        dataset,
        seed,
        dictionary,
        atom_key,
        pos_key,
        is_train,
        args,
    ):
        super().__init__(
            dataset,
            seed,
            dictionary,
            atom_key,
            pos_key,
            is_train,
            args,
        )

    def get_basic_feat(self, data):
        feat = RawAtomFeature.init_from_mol_ene(
            data[self.pos_key],
            data[self.atom_key],
        )
        self.add_atom_and_mol_targets(data, feat)
        return feat

    def get_grid_feat(self, data, atom_feat):

        assert atom_feat.data_type == "molecule_energy"
        atom_grid_pos, atom_grid_xyz, atom_feat, cutoff_atom_count = self.make_grid(
            atom_feat,
        )
        all_atom_grid_feat_withH = AtomGridFeature(
            atom_feat.atom_type,
            atom_grid_pos,
            atom_grid_xyz,
            self.dictionary,
            atom_feats=atom_feat.atom_feats,
            mol_feats=atom_feat.mol_feats,
        )
        # avoid reuse
        del atom_grid_pos, atom_grid_xyz
        all_atom_grid_feat = self.H_prob_strategy(all_atom_grid_feat_withH)
        raw_atom_count = all_atom_grid_feat.atom_pos.shape[0]

        # construct tree
        decoder_results = self.construct_tree_for_decoder(all_atom_grid_feat)

        decoder_cond = molecule_energy_data_utils.get_molecule_energy_cond(
            self.args,
            data,
            self.is_train,
            self.dictionary,
            self.xyz_null_id,
        )
        # 将条件特征与其他特征置于同一个tree和同一个space种，但条件特征不参与损失计算
        final_decoder_feature = self.concat_tree_feats(
            [decoder_cond, decoder_results],
            trees=[0, 0],
            spaces=[0, 0],
            tree_losses=[0, 1],
            keys=decoder_results.keys(),
        )
        final_decoder_feature["cutoff_atom_count"] = cutoff_atom_count
        final_decoder_feature["raw_atom_count"] = raw_atom_count

        final_decoder_feature = self.wrap_decoder_features(final_decoder_feature)

        if self.args.data_type == "molecule_energy":
            final_decoder_feature["energy"] = torch.from_numpy(
                decoder_cond["energy"]
            ).float()



        return final_decoder_feature
