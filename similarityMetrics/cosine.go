package similarityMetrics

import (
	"math"
	"sort"
)

type CosineConfidence struct{}

func (CosineConfidence) GetConfidenceScore(queryVector []float64, instances [][]float64) []Confidence {
	confidence := make([]Confidence, len(instances))
	for i, instance := range instances {
		confidence[i].ConfidenceScore = cosineSimilarity(queryVector, instance)
		confidence[i].Index = i
	}

	sort.SliceStable(confidence, func(i, j int) bool {
		return confidence[i].ConfidenceScore > confidence[j].ConfidenceScore
	})

	return confidence
}

func cosineSimilarity(vec1, vec2 []float64) float64 {
	dotProduct := 0.0
	magnitude1 := 0.0
	magnitude2 := 0.0
	for i := range vec1 {
		dotProduct += vec1[i] * vec2[i]
		magnitude1 += vec1[i] * vec1[i]
		magnitude2 += vec2[i] * vec2[i]
	}
	magnitude1 = math.Sqrt(magnitude1)
	magnitude2 = math.Sqrt(magnitude2)
	return dotProduct / (magnitude1 * magnitude2)
}
