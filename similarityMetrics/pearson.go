package similarityMetrics

import (
	"math"
	"sort"
)

type PearsonConfidence struct{}

func (PearsonConfidence) GetConfidenceScore(queryVector []float64, instances [][]float64) []Confidence {
	confidence := make([]Confidence, len(instances))
	for i, instance := range instances {
		confidence[i].ConfidenceScore = pearsonCorrelation(queryVector, instance)
		confidence[i].Index = i
	}

	sort.SliceStable(confidence, func(i, j int) bool {
		return confidence[i].ConfidenceScore > confidence[j].ConfidenceScore
	})

	return confidence
}

func pearsonCorrelation(vec1, vec2 []float64) float64 {
	sumX, sumY, sumXY, sumXSquare, sumYSquare := 0.0, 0.0, 0.0, 0.0, 0.0
	n := len(vec1)
	for i := range vec1 {
		sumX += vec1[i]
		sumY += vec2[i]
		sumXY += vec1[i] * vec2[i]
		sumXSquare += math.Pow(vec1[i], 2)
		sumYSquare += math.Pow(vec2[i], 2)
	}
	numerator := (float64(n)*sumXY - sumX*sumY)
	denominator := math.Sqrt((float64(n)*sumXSquare - math.Pow(sumX, 2)) * (float64(n)*sumYSquare - math.Pow(sumY, 2)))
	return numerator / denominator
}
