from tensor2tensor import problems
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import metrics_hook
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics
import pandas as pd

#from ADLS_access import access_file_from_directory


@registry.register_problem
class SigTranslator(text_problems.Text2TextProblem):
    """Predict next line of poetry from the last line. From Gutenberg texts."""

    @property
    def approx_vocab_size(self):
        return 2 ** 13 * 2 * 2  # ~32k

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 20% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def vocab_type(self):
        # We can use different types of vocabularies, `VocabType.CHARACTER`,
        # `VocabType.SUBWORD` and `VocabType.TOKEN`.
        #
        # SUBWORD and CHARACTER are fully invertible -- but SUBWORD provides a good
        # tradeoff between CHARACTER and TOKEN.
        return text_problems.VocabType.SUBWORD

    def eval_metrics(self):
        return [
            metrics.Metrics.ACC,
            metrics.Metrics.ACC_TOP5,
            metrics.Metrics.ACC_PER_SEQ,
            metrics.Metrics.NEG_LOG_PERPLEXITY,
            metrics.Metrics.ROUGE_L_F,
            metrics.Metrics.APPROX_BLEU,

        ]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        #data_path = 'translation/dataset/roa_dosage/OR_TABS_processed_data/'

        #sig_data = access_file_from_directory("sig-erx", data_path, "OR_TABS_Final_Data_Training_Full_P_v2.csv")
        sig_data = pd.read_csv('/data/home/users/pnadim64/notebooks/Raju/Translation/T2t/Data/OR_OTH_Train_data_Top3_V4.0.csv',usecols = ['Standardized_SIG', 'IC_Pharmacist_SIG'])

        #sig_data = sig_data.head(100000)
        sig_data = sig_data[['Standardized_SIG', 'IC_Pharmacist_SIG']].drop_duplicates()

        print("******************************Data size:****************************************", sig_data.shape)

        #sig_data = sig_data[['Standardized_SIG', 'rx_sig']]

        #sig_data = process_data(sig_data)

        

        for sig in range(len(sig_data)):
            yield {
                "inputs": sig_data.Standardized_SIG.iloc[sig],
                "targets": sig_data.IC_Pharmacist_SIG.iloc[sig],
            }
