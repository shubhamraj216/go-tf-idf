package ngrams

import "regexp"

func CalculateNGrams(str string, n int) []string {
	n = 1
	// Remove punctuation from the string
	reg, _ := regexp.Compile(`[,-./]|\sBD`)
	str = reg.ReplaceAllString(str, "")

	// Generate zip of ngrams (n defined in function argument)
	var result []string
	for i := 0; i < len(str)-n+1; i++ {
		result = append(result, str[i:i+n])
	}

	return result
}
