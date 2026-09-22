import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import torchaudio

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

class Cluster():
    def __init__(self, label, dist_metric=cdist):
        """ Cluster class
        It is used to segments clustering.
        Parameters
        ----------
        label: string
            name of cluster
        dist_metric: optional
            distance metric_bak to compute distance
            between clusters and models
        """
        self.label = label  # cluster name
        self.representation = np.zeros(256)  # cluster embedding
        self.dist_metric = cdist  # distance metric_bak
        self.embeddings = []  # embedding of segments in this cluster
        self.indices = []
        self.durations = []  # segements in this cluster
        self.distances = {}
        self.segments = []
        self.confidences = []
        self.idx_generator = 0
        self.labels = []

    def update(self, data, dist = 0, thresholdMerge = 100, seg_duration_threshold = 0, allow_center_update=True):
        """
        update the cluster:
            add new segment embedding
            update the cluster embedding
            add segment to cluster
            add distances between model and cluster
        """
        # self.embeddings.append(np.zeros(256))
        self.embeddings.append(data['embedding'])
        self.segments.append((data['start'], data['end']))
        self.indices.append(-1)
        self.labels.append(-1)
        self.durations.append(data['duration'])
        self.confidences.append(data['confidence'])
        if dist > thresholdMerge or data['duration'] < seg_duration_threshold or data['confidence'] < 0.2 or not allow_center_update:
            return
        else:
            self.representation += normalize(data['embedding']) * data['duration'] * data['confidence']
        return


    def updateCluster(self, data, dist = 0, thresholdMerge = 100, seg_duration_threshold = 0, allow_center_update = True):
        """
        update the cluster:
            add new segment embedding
            update the cluster embedding
            add segment to cluster
            add distances between model and cluster
        """
        self.embeddings.append(data['embedding'])
        self.segments.append((data['start'], data['end']))
        self.indices.append(self.idx_generator)
        self.labels.append(-1)
        self.idx_generator += 1
        self.durations.append(data['duration'])
        self.confidences.append(data['confidence'])
        is_high_quality = (data['duration'] >= seg_duration_threshold) and (data['confidence'] >= 0.2)
        if not hasattr(self, 'is_anchored'):
            self.is_anchored = False
        new_vector = normalize(data['embedding']) * data['confidence'] * data['duration']
        if not self.is_anchored:
            if is_high_quality:
                # 迎来了第一个高质量片段！直接将其作为绝对中心，不考虑之前的烂数据
                self.representation = new_vector
                self.is_anchored = True
            else:
                # 依然是低质量片段。给予一个临时表征用于距离计算，但不使用 alpha 固化历史
                if np.linalg.norm(self.representation) > 0:
                    # 简单相加归一化作为临时中心，防止单次极端噪音导致完全偏移
                    self.representation = normalize(self.representation + new_vector) 
                else:
                    self.representation = new_vector
            return
        if dist > thresholdMerge or data['duration'] < seg_duration_threshold or data['confidence'] < 0.2 or not allow_center_update:
            return
        else:
            self.representation += normalize(data['embedding']) * data['duration'] * data['confidence']
        return

    def updateFilter(self, data, dist, thresholdMerge, allow_center_update):
        if max(self.indices) > 120:
            self.update(data, dist, thresholdMerge, 85, allow_center_update)
            return
        elif max(self.indices) < 5:
            self.updateCluster(data, dist, thresholdMerge, 85, allow_center_update)
        else:
            self.embeddings.append(data['embedding'])
            self.durations.append(data['duration'])
            self.indices.append(self.idx_generator)
            self.labels.append(-1)
            self.idx_generator = max(self.indices) + 1
            self.segments.append((data['start'], data['end']))
            self.confidences.append(data['confidence'])
        return


class OnlineClustering:
    

    def __init__(self,
                threshold=0.35,
                threshold_filter=0.35,
                online_hac_threshold2 = 0.45,
                threshold_online_spknum_10 = 0.35,
                threshold_filter_online_spknum_6 = 0.33,
                cluster_update_len=90,
                cluster_update_len2 = 70,
                generator_method='int'):

        self.high_size_filter = 120
        self.low_size_filter = 5
        self.cluster_update_len = cluster_update_len
        self.cluster_update_len2 = cluster_update_len2
        self.threshold = threshold
        self.threshold_filter = threshold_filter
        self.online_hac_threshold2 = online_hac_threshold2
        self.threshold_online_spknum_10 = threshold_online_spknum_10
        self.threshold_filter_online_spknum_6 = threshold_filter_online_spknum_6
        self.counter = 0
        self.clusters = []  # store the current clusters
        self.generator_method = generator_method  # generate cluster names

        if self.generator_method == 'string':
            from pyannote.core.utils.generators import string_generator
            self.generator = string_generator()
        elif self.generator_method == 'int':
            from pyannote.core.utils.generators import int_generator
            self.generator = int_generator()


    def getLabels(self):
        """
        returns all the cluster labels
        """
        return [cluster.label for cluster in self.clusters]

    def addCluster(self, data):
        """
        create a new cluster
        """
        label = next(self.generator)
        cluster = Cluster(label)
        cluster.updateCluster(data)
        self.clusters.append(cluster)
        self.counter += 1
        return label

    def get_representation(self, embeddings, durations, confidences):
        res = np.zeros(256)
        for embedding, duration, conf in zip(embeddings, durations, confidences):
            res += normalize(embedding) * duration * conf
        return res

    def get_duration_of_label(self, durations, labels):
        labels_uniq = set(labels)
        res = []
        for label in labels_uniq:
            res.append((label, sum(durations[labels == label])))
        res.sort(key=lambda item: item[1], reverse=True)
        return res

    
    def computeDistances(self, data):
        if not self.clusters:
            return [1.0]
            
        # 将所有簇的 representation 堆叠成一个矩阵 (K x 256)
        cluster_reps = np.array([c.representation for c in self.clusters])
        feature = np.expand_dims(data['embedding'], axis=0)
        
        # 使用 cdist 一次性计算输入与所有簇的余弦距离
        # 返回结果是一个一维数组，直接 tolist()
        distances = cdist(feature, cluster_reps, metric='cosine')[0]
        return distances.tolist()

    def split_cluster(self, cluster, threshold):
        if not cluster.indices: # 防止空列表导致 max() 报错
            return
            
        if max(cluster.indices) < self.low_size_filter or max(cluster.indices) > self.high_size_filter:
            return

        # 2. 生成过滤掩码 (duration > 80)
        durations_np = np.array(cluster.durations)
        selected_mask = durations_np > 80

        # 3. 提取过滤后的数据
        embeddings = np.array(cluster.embeddings)[selected_mask]
        
        # 如果过滤后没有任何数据留下，清空 cluster 并返回
        if len(embeddings) == 0:
            return

        durations = durations_np[selected_mask]
        segments = np.array(cluster.segments)[selected_mask]
        confidences = np.array(cluster.confidences)[selected_mask]
        indices = np.array(cluster.indices)[selected_mask]

        # 4. 关键修改：直接用过滤后的数据覆盖 cluster 原有的属性，实现“只保留 > 80 的元素”
        cluster.embeddings = embeddings.tolist()
        cluster.durations = durations.tolist()
        cluster.segments = segments.tolist()
        cluster.confidences = confidences.tolist()
        cluster.indices = indices.tolist()
        cluster.labels = [-1] * len(embeddings)

        if len(embeddings) < 2:
            if len(embeddings) == 1:
                cluster.representation = normalize(embeddings[0])
            return

        # 5. 计算距离矩阵并进行层次聚类
        distance_matrix = cosine_distances(embeddings)
        algo_hac = AgglomerativeClustering(
            n_clusters=None, 
            linkage='average', 
            distance_threshold=threshold, 
            metric='precomputed'
        )
        
        try:
            labels = algo_hac.fit(distance_matrix).labels_
        except Exception as e:
            print(f"Clustering failed: {e}")
            return

        # 6. 因为底层数据已经过滤过了，labels 数组的长度和 cluster 目前的元素长度完全一致
        # 直接赋值即可，不再需要 -1 占位！
        cluster.labels = labels.tolist()

        # 7. 检查是否成功分出了多个簇
        labels_uniq = set(labels)
        # if len(labels_uniq) <= 1:
        #     return

        # 8. 获取主导的标签，并更新 cluster 的 representation
        dominant_label = self.get_duration_of_label(durations, labels)[0][0]
        
        filtered_mask = (labels == dominant_label)
        filtered_embeddings = embeddings[filtered_mask]
        filtered_durations = durations[filtered_mask]
        filtered_confidences = confidences[filtered_mask]

        cluster.representation = self.get_representation(
            filtered_embeddings, 
            filtered_durations, 
            filtered_confidences
        )
        return

    def JudgeAddNewCluster(self, dis, data):
        max_cluster_num_optimal = 6
        max_cluster_num_accepetable = 10
        min_cluster_num_unreliable = max_cluster_num_accepetable + 1

        if data['confidence'] < 0.5:
            return False

        if self.counter <= max_cluster_num_optimal:
            if (data['duration'] > self.cluster_update_len and dis > self.threshold) or \
                (data['duration'] > self.cluster_update_len2 and dis > self.online_hac_threshold2):
                return True
        elif self.counter <= max_cluster_num_accepetable and self.counter > max_cluster_num_optimal:
            if (data['duration'] > self.cluster_update_len and dis > self.threshold):
                return True
        elif self.counter > max_cluster_num_accepetable:
            if (data['duration'] > self.cluster_update_len and dis > self.threshold_online_spknum_10):
                return True
        else:
            return False

    def merge(self):
        i = 0
        while i < len(self.clusters)-1:
            cluster1 = self.clusters[i]
            # 使用 labels 而不是 indices
            if not cluster1.labels:
                i += 1
                continue
            dominant_label1, duration1 = self.get_duration_of_label(np.array(cluster1.durations), cluster1.labels)[0]
            cluster1_dominant_size = sum(np.array(cluster1.labels) == dominant_label1)
            j = i + 1
            while j < len(self.clusters):
                cluster2 = self.clusters[j]
                if not cluster2.labels:
                    j += 1
                    continue
                dominant_label2, duration2 = self.get_duration_of_label(np.array(cluster2.durations), cluster2.labels)[0]
                cluster2_dominant_size = sum(np.array(cluster2.labels) == dominant_label2)
                if duration1 > 800 and duration2 > 800:
                    dis = cluster1.dist_metric(np.expand_dims(cluster1.representation, axis=0), \
                                                np.expand_dims(cluster2.representation, axis=0), metric='cosine')[0]
                    if dis < 0.25:
                        # 根据持续时间大小决定合并方向
                        if duration1 > duration2:
                            # 将cluster2合并到cluster1
                            # 处理label冲突：重新编号cluster2的labels
                            max_label = max(cluster1.labels) if cluster1.labels else -1
                            cluster2_relabeled = [l + max_label + 1 if l != -1 else -1 for l in cluster2.labels]

                            # 合并所有数据
                            cluster1.embeddings.extend(cluster2.embeddings)
                            cluster1.durations.extend(cluster2.durations)
                            cluster1.segments.extend(cluster2.segments)
                            cluster1.confidences.extend(cluster2.confidences)
                            cluster1.indices.extend(cluster2.indices)
                            cluster1.labels.extend(cluster2_relabeled)
                            # 更新idx_generator
                            cluster1.idx_generator = max(cluster1.indices) + 1

                            # 更新representation
                            cluster1.representation = cluster1.representation * duration1 \
                                + cluster2.representation * duration2
                            cluster1.representation = normalize(cluster1.representation)

                            # 删除cluster2
                            del self.clusters[j]
                            self.counter -= 1
                            if j == len(self.clusters):
                                break
                            else:
                                continue
                        else:
                            # 将cluster1合并到cluster2
                            # 处理label冲突：重新编号cluster1的labels
                            max_label = max(cluster2.labels) if cluster2.labels else -1
                            cluster1_relabeled = [l + max_label + 1 if l != -1 else -1 for l in cluster1.labels]

                            # 合并所有数据到cluster2
                            cluster2.embeddings.extend(cluster1.embeddings)
                            cluster2.durations.extend(cluster1.durations)
                            cluster2.segments.extend(cluster1.segments)
                            cluster2.confidences.extend(cluster1.confidences)
                            cluster2.indices.extend(cluster1.indices)
                            cluster2.labels.extend(cluster1_relabeled)
                            # 更新idx_generator
                            cluster2.idx_generator = max(cluster2.indices) + 1

                            # 更新representation
                            cluster2.representation = cluster1.representation * duration1 \
                                + cluster2.representation * duration2
                            cluster2.representation = normalize(cluster2.representation)

                            # 删除cluster1
                            del self.clusters[i]
                            self.counter -= 1
                            # 调整i，因为删除了当前元素
                            i -= 1
                            break
                    else:
                        j += 1
                else:
                    j += 1
            i += 1
        return

    def update(self, data):
        
        if self.counter == 0:
            label = self.addCluster(data)
            return label

        distances = self.computeDistances(data)
        if self.JudgeAddNewCluster(min(distances), data):
            label = self.addCluster(data)
        else:
            sorted_indices = np.argsort(distances)
            indice = sorted_indices[0]
            allow_center_update = True
            if len(distances) > 1 and self.counter > 6:
                if distances[sorted_indices[1]] - distances[indice] < 0.05:
                    allow_center_update = False
            to_update_cluster = self.clusters[indice]
            to_update_cluster.updateFilter(data, min(distances), self.threshold, allow_center_update)
            if self.counter < 6:
                self.split_cluster(to_update_cluster, self.threshold_filter)
            else:
                self.split_cluster(to_update_cluster, self.threshold_filter_online_spknum_6)

            
            label = to_update_cluster.label
        #calculate centriod
        if self.counter <= 2:
            return label
        self.merge()
        return label

    @staticmethod
    def get_duration(file) -> float:
        audio = str(file)

        info = torchaudio.info(audio)
        return info.num_frames / info.sample_rate

    def empty(self):
        if len(self.clusters) == 0:
            return True
        return False


