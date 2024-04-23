package tfidf

type TfidfVectorizer struct {
	Vocabulary  map[string]int
	NgramFunc   func(string, int) []string
	NgramLength int
	MinDF       int
}
