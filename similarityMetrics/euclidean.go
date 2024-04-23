package similarityMetrics

import (
	"math"
	"sort"
)

type EuclideanConfidence struct{}

func (EuclideanConfidence) GetConfidenceScore(queryVector []float64, instances [][]float64) []Confidence {
	confidence := make([]Confidence, len(instances))
	for i, instance := range instances {
		confidence[i].ConfidenceScore = 1.0 - euclideanDistance(queryVector, instance)
		confidence[i].Index = i
	}

	sort.SliceStable(confidence, func(i, j int) bool {
		return confidence[i].ConfidenceScore > confidence[j].ConfidenceScore
	})

	return confidence
}

func euclideanDistance(vec1, vec2 []float64) float64 {
	sum := 0.0
	for i := range vec1 {
		sum += math.Pow(vec1[i]-vec2[i], 2)
	}
	return math.Sqrt(sum)
}
