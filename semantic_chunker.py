import copy
import re
from typing import Dict, List, Literal, Optional, Tuple, cast
import numpy as np
from embeddings import Embeddings

def cosine_similarity(vectors1: List[np.ndarray], vectors2: List[np.ndarray]) -> np.ndarray:
    dot_product = np.dot(vectors1, np.transpose(vectors2))
    norm1 = np.linalg.norm(vectors1, axis=1)
    norm2 = np.linalg.norm(vectors2, axis=1)
    return dot_product / np.outer(norm1, norm2)

def combine_sentences(sentences: List[dict], buffer_size: int = 1) -> List[dict]:
    for i in range(len(sentences)):
        combined_sentence = ""
        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sentence += sentences[j]["sentence"] + " "
        combined_sentence += sentences[i]["sentence"]
        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                combined_sentence += " " + sentences[j]["sentence"]
        sentences[i]["combined_sentence"] = combined_sentence
    return sentences

def calculate_cosine_distances(sentences: List[dict]) -> Tuple[List[float], List[dict]]:
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]["combined_sentence_embedding"]
        embedding_next = sentences[i + 1]["combined_sentence_embedding"]
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        distance = 1 - similarity
        distances.append(distance)
        sentences[i]["distance_to_next"] = distance
    return distances, sentences

BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile", "gradient"]
BREAKPOINT_DEFAULTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
    "gradient": 95,
}

class SemanticChunker:
    def __init__(
        self,
        embeddings: Embeddings,
        buffer_size: int = 1,
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        number_of_chunks: Optional[int] = None,
        sentence_split_regex: str = r"(?<=[.?!])\s+",
    ):
        self.embeddings = embeddings
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.number_of_chunks = number_of_chunks
        self.sentence_split_regex = sentence_split_regex
        if breakpoint_threshold_amount is None:
            self.breakpoint_threshold_amount = BREAKPOINT_DEFAULTS[breakpoint_threshold_type]
        else:
            self.breakpoint_threshold_amount = breakpoint_threshold_amount

    def _calculate_breakpoint_threshold(
        self, distances: List[float]
    ) -> Tuple[float, List[float]]:
        if self.breakpoint_threshold_type == "percentile":
            return cast(
                float,
                np.percentile(distances, self.breakpoint_threshold_amount),
            ), distances
        elif self.breakpoint_threshold_type == "standard_deviation":
            return cast(
                float,
                np.mean(distances)
                + self.breakpoint_threshold_amount * np.std(distances),
            ), distances
        elif self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1
            return np.mean(distances) + self.breakpoint_threshold_amount * iqr, distances
        elif self.breakpoint_threshold_type == "gradient":
            distance_gradient = np.gradient(distances, range(0, len(distances)))
            return cast(
                float,
                np.percentile(distance_gradient, self.breakpoint_threshold_amount),
            ), distance_gradient
        else:
            raise ValueError(
                f"Got unexpected `breakpoint_threshold_type`: "
                f"{self.breakpoint_threshold_type}"
            )

    def _threshold_from_clusters(self, distances: List[float]) -> float:
        if self.number_of_chunks is None:
            raise ValueError(
                "This should never be called if `number_of_chunks` is None."
            )
        x1, y1 = len(distances), 0.0
        x2, y2 = 1.0, 100.0
        x = max(min(self.number_of_chunks, x1), x2)
        y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)
        y = min(max(y, 0), 100)
        return cast(float, np.percentile(distances, y))

    def _calculate_sentence_distances(
        self, single_sentences_list: List[str]
    ) -> Tuple[List[float], List[dict]]:
        _sentences = [{"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)]
        sentences = combine_sentences(_sentences, self.buffer_size)
        embeddings = self.embeddings.embed_documents(
            [x["combined_sentence"] for x in sentences]
        )
        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[i]
        return calculate_cosine_distances(sentences)

    def split_text(self, text: str) -> List[str]:
        single_sentences_list = re.split(self.sentence_split_regex, text)
        if len(single_sentences_list) == 1:
            return single_sentences_list
        distances, sentences = self._calculate_sentence_distances(single_sentences_list)
        if self.number_of_chunks is not None:
            breakpoint_distance_threshold = self._threshold_from_clusters(distances)
            breakpoint_array = distances
        else:
            breakpoint_distance_threshold, breakpoint_array = self._calculate_breakpoint_threshold(distances)
        indices_above_thresh = [i for i, x in enumerate(breakpoint_array) if x > breakpoint_distance_threshold]
        chunks = []
        start_index = 0
        for index in indices_above_thresh:
            end_index = index
            group = sentences[start_index : end_index + 1]
            combined_text = " ".join([d["sentence"] for d in group])
            chunks.append(combined_text)
            start_index = index + 1
        if start_index < len(sentences):
            combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
            chunks.append(combined_text)
        return chunks
