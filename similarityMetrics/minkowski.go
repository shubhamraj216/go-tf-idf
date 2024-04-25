package similarityMetrics

import (
	"math"
	"sort"
)

type MinkowskiConfidence struct{}

func (MinkowskiConfidence) GetConfidenceScore(queryVector []float64, instances [][]float64) []Confidence {
	confidence := make([]Confidence, len(instances))
	for i, instance := range instances {
		confidence[i].ConfidenceScore = minkowskiDistance(queryVector, instance, 2)
		confidence[i].Index = i
	}

	sort.SliceStable(confidence, func(i, j int) bool {
		return confidence[i].ConfidenceScore < confidence[j].ConfidenceScore
	})

	return confidence
}

func minkowskiDistance(vec1, vec2 []float64, p float64) float64 {
	sum := 0.0
	for i := range vec1 {
		sum += math.Pow(math.Abs(vec1[i]-vec2[i]), p)
	}
	return math.Pow(sum, 1/p)
}
