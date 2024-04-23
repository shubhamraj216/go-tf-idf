package go_tf_idf

import (
	"fmt"
	"github.com/shubhamraj216/go-tf-idf/similarityMetrics"
)

func getClosestMatch() {
	original := "This is a nest"
	lookup := []string{"This is a test", "Test is good", "who is this"}

	indices, confidence, _ := CalculateNearestMatch(original, lookup, 3, 3, 1, similarityMetrics.COSINE_SIMILARITY)

	fmt.Println(indices, confidence) // [0 2 1] [0.987241120712647 0.85202864568461 0.7889320586105298]
}
