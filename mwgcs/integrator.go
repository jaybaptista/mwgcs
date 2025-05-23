package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
)

const dim int = 3
const phaseSpaceSize int = 2 * dim
const particleCount int = 1
const h float64 = 0.01

// Deriv is a type for the derivative function
type Deriv func(w [phaseSpaceSize][particleCount]float64, t float64) [phaseSpaceSize][particleCount]float64

func addArrays(arrays ...[phaseSpaceSize][particleCount]float64) [phaseSpaceSize][particleCount]float64 {
	var result [phaseSpaceSize][particleCount]float64
	for _, array := range arrays {
		for i := range array {
			for j := range array[i] {
				result[i][j] += array[i][j]
			}
		}
	}
	return result
}

func scalarMultiply(scalar float64, array [phaseSpaceSize][particleCount]float64) [phaseSpaceSize][particleCount]float64 {
	var result [phaseSpaceSize][particleCount]float64
	for i := range array {
		for j := range array[i] {
			result[i][j] = scalar * array[i][j]
		}
	}
	return result
}

func quadratureSum(array [phaseSpaceSize][particleCount]float64) float64 {
	var sum float64
	for i := range array {
		for j := range array[i] {
			sum += array[i][j] * array[i][j]
		}
	}
	// square root the sum
	sum = math.Sqrt(sum)
	return sum
}

func RK4Step(w [phaseSpaceSize][particleCount]float64, t float64, h float64, deriv Deriv) [phaseSpaceSize][particleCount]float64 {
	// Deriv is a type for the derivative function
	k1 := deriv(w, t)
	k2 := deriv(addArrays(w, scalarMultiply(h/2, k1)), t+h/2)
	k3 := deriv(addArrays(w, scalarMultiply(h/2, k2)), t+h/2)
	k4 := deriv(addArrays(w, scalarMultiply(h, k3)), t+h)

	return addArrays(w, scalarMultiply(h/6, addArrays(k1, scalarMultiply(2, k2), scalarMultiply(2, k3), k4)))
}

func derivative(w [phaseSpaceSize][particleCount]float64, t float64) [phaseSpaceSize][particleCount]float64 {
	var result [phaseSpaceSize][particleCount]float64

	velocityIndex := int(phaseSpaceSize / 2)

	// velocities are the first half of the phase space
	// loop over particles

	for j := 0; j < particleCount; j++ {
		for i := 0; i < velocityIndex; i++ {
			result[i][j] = w[velocityIndex+i][j]
		}

		// calculate the accelerations
		for i := velocityIndex; i < phaseSpaceSize; i++ {
			result[i][j] = -1 / (w[i][j] * w[i][j])
		}
	}

	return result
}

func createRandomInitialConditions(mu float64, sigma float64) [phaseSpaceSize][particleCount]float64 {
	var w [phaseSpaceSize][particleCount]float64
	for i := range w {
		for j := range w[i] {
			w[i][j] = rand.NormFloat64()*sigma + mu
		}
	}
	return w
}

func main() {
	// Deriv is a type for the derivative function
	var w0 [phaseSpaceSize][particleCount]float64 = createRandomInitialConditions(3, .2)
	var t0 float64 = 0.0
	var tf float64 = 10.0

	var stepCount int = int((tf - t0) / h)

	var ws = make([][phaseSpaceSize][particleCount]float64, stepCount)

	ws[0] = w0

	for i := 1; i < stepCount; i++ {
		ws[i] = RK4Step(ws[i-1], t0+float64(i)*h, h, derivative)
	}

	// save the results as a csv

	file, err := os.Create("results.csv")
	if err != nil {
		panic(err)
	}

	defer file.Close()

	writer := bufio.NewWriter(file)

	for i := range ws {
		for j := range ws[i] {
			for k := range ws[i][j] {
				writer.WriteString(fmt.Sprintf("%f", ws[i][j][k]))
				writer.WriteString(",")
			}
			writer.WriteString("\n")
		}
	}
}
