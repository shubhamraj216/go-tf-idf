package similarityMetrics

type ConfidenceScore interface {
	GetConfidenceScore([]float64, [][]float64) []Confidence
}
