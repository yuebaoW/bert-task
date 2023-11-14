import sys
sys.path.append("..")
# 注意或者sys.path.append('.')或者直接引入包名
# sys.path.append('path/项目包名directory')（添加模块完整路径）参考第二种方法

from config.reg_config import train_parameters

from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator


pre_train_root_dir = train_parameters.pre_train_root_dir
dict_path = pre_train_root_dir + '/vocab.txt'


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=train_parameters.maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels

                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

