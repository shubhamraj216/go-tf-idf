package similarityMetrics

import (
	"math"
	"sort"
)

type ManhattanConfidence struct{}

func (ManhattanConfidence) GetConfidenceScore(queryVector []float64, instances [][]float64) []Confidence {
	confidence := make([]Confidence, len(instances))
	for i, instance := range instances {
		confidence[i].ConfidenceScore = manhattanDistance(queryVector, instance)
		confidence[i].Index = i
	}

	sort.SliceStable(confidence, func(i, j int) bool {
		return confidence[i].ConfidenceScore < confidence[j].ConfidenceScore
	})

	return confidence
}

func manhattanDistance(vec1, vec2 []float64) float64 {
	sum := 0.0
	for i := range vec1 {
		sum += math.Abs(vec1[i] - vec2[i])
	}
	return sum
}
