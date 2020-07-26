import tensorflow as tf
import importlib


class LookupTables(object):
    """docstring for Inference"""

    def __init__(self, wnd_conf):
        super(LookupTables, self).__init__()
        # data member: self.wnd_conf
        self.wnd_conf = wnd_conf

        self.embLookupTables = {}
        embedding_list = []
        embedding_list.extend(wnd_conf.embedding_list)
        embedding_list.extend(wnd_conf.embedding_list_bias)

        for emb in embedding_list:
            emb_name = emb[0]
            idstype = importlib.import_module('.%s' % emb_name, 'idtables')
            id_size = int(emb[1])
            id_list_len = len(idstype.ID_TABLES[emb_name])
            bucket_size = id_size - id_list_len
            print("read idtables : %s, size = %s" % (emb_name, id_list_len))

            if emb_name not in self.embLookupTables:
                self.embLookupTables[emb_name] = tf.contrib.lookup.index_table_from_tensor( \
                    mapping=tf.constant(idstype.ID_TABLES[emb_name]), num_oov_buckets=bucket_size, default_value=0)

        self.featureLookupTables = {}
        for emb in embedding_list:
            emb_name = emb[0]
            feature_name = emb[3]
            if feature_name not in self.featureLookupTables:
                self.featureLookupTables[feature_name] = self.embLookupTables[emb_name]

    def transform_id2index(self, features):
        for key, value in features.items():
            if key in self.featureLookupTables:
                raw_feature = features[key]
                features[key] = tf.SparseTensor(indices=raw_feature.indices, \
                                                values=self.featureLookupTables[key].lookup(raw_feature.values),
                                                dense_shape=raw_feature.dense_shape)
            else:
                print("don't find %s in tables" % key)

    def inf_transform(self, id_name, ids):
        if id_name in self.featureLookupTables:
            index = self.featureLookupTables[id_name].lookup(ids)
            return index
        else:
            print("don't find %s in tables" % id_name)

    def lookup_embedding(self, emb_name, ids):
        if emb_name in self.embLookupTables:
            index = self.embLookupTables[emb_name].lookup(ids)
            return index
        else:
            print("don't find %s in tables" % emb_name)
