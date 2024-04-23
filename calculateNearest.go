package go_tf_idf

import (
	"errors"
	"github.com/shubhamraj216/go-tf-idf/knn"
	"github.com/shubhamraj216/go-tf-idf/ngrams"
	"github.com/shubhamraj216/go-tf-idf/similarityMetrics"
	"github.com/shubhamraj216/go-tf-idf/tfidf"
	"strings"
)

/*
* CalculateNearestMatch calculates the nearest match of the original string in the lookup list

  - @param original string - the original string to find the nearest match

  - @param lookup []string - the list of strings to compare the original string against

  - @param kMatches int - the number of matches to return - default returns all matches

  - @param ngramLength int - specifies the size of the sliding window used to extract n-grams from the text - defaults to 3.

  - @param minDF int - the minimum document frequency - defaults to 1.
    Used to specify the minimum document frequency for a term to be included in the vocabulary.
    Setting minDF=1 means that a term must appear in at least one document to be included in the vocabulary.

  - @param similarityCheckMetric similarityMetrics.SIMILARITY_CHECK_METRIC - the similarity check metric to use - defaults to COSINE_SIMILARITY

  - @return []int - the indices of the nearest matches

  - @return []float64 - the confidence scores of the nearest matches

  - @return error - an error if the lookup list is empty or the original string is empty
*/
func CalculateNearestMatch(original string, lookup []string, kMatches, ngramLength, minDF int,
	similarityCheckMetric similarityMetrics.SIMILARITY_CHECK_METRIC) ([]int, []float64, error) {

	if len(lookup) == 0 {
		return nil, nil, errors.New("Lookup list is empty")
	}
	if len(original) == 0 {
		return nil, nil, errors.New("Original string is empty")
	}

	if kMatches == 0 {
		kMatches = len(lookup)
	}

	if ngramLength == 0 {
		ngramLength = 3
	}

	if minDF == 0 {
		minDF = 1
	}

	original = strings.ToLower(original)

	for idx, str := range lookup {
		lookup[idx] = strings.ToLower(str)
	}

	vectorizer := tfidf.NewTfidfVectorizer(ngrams.CalculateNGrams, ngramLength, minDF)

	tfIdfLookup := vectorizer.FitTransform(lookup)
	tfidfOriginal := vectorizer.Transform(original)

	nbrs := knn.NewNearestNeighbors(kMatches, similarityCheckMetric)
	nbrs.Fit(tfIdfLookup)

	indices, confidence := nbrs.KNeighbors(tfidfOriginal)

	return indices, confidence, nil
}
