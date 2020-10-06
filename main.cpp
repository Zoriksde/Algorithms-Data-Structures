#include <iostream>
#include <vector>
#include <stack>
#include <queue>

/* Zaimplementuj Min - Heap [ ZROBIONE ] */ 
/* Zaimplementuj QuickSort ( rekurencyjne z osobn¹ funkcj¹ partycjonuj¹c¹, oraz iteracyjnie )  [ ZROBIONE ] */
/* Zaimplementuj BubbleSort ( z usprawnieniami ), przeanalizuj jego z³o¿onoœæ czasow¹ i pamiêciow¹ [ ZROBIONE ]  */
/* Zaimplementuj RadixSort, dla liczb zapisanych w systemie dziesiêtnym [ ZROBIONE ] */
/* Zaimplementuj SelectionSort, oraz InsertionSort oraz porównaj je pod wzglêdem z³o¿onoœci czasowej [ ZROBIONE ] */
/* Zaimplementuj Heap Sort, pisz¹c w³asne funkcje konstrukcji takiego kopca, zapisz z³o¿onoœci czasowe 
operacji : construct, heapdown  [ ZROBIONE ] */
/* Zaimplementuj Algorytm Hoare ( quickselect ), z algorytmem partycjonuj¹cym Hoare ( rekurencyjnie, iteracyjnie  [ ZROBIONE ] */
/* Zaimplementuj Algorytm Bluma Floyda Pratta Rivesta Tarjana ( osoba funkcja do mediany i partycjonowania )  [ ZROBIONE ]*/
/* Zaimplementuj Algorytm Binary Search ( rekurencyjnie oraz iteracyjnie ) [ ZROBIONE ] */
/* Zaimplementuj QuickSort ( rekurencyjnie z granic¹ ), oraz MergeSort ( rekurencyjnie ) [ ZROBIONE ] */
/* ====================================================================================================== */
/* Zaimplementuj ShellSort ( dowolny algorytm licz¹cy dystans ) [ ZROBIONE ] */
/* Zaimplementuj Algorytm Euklidesa ( rekurencyjnie oraz iteracyjnie ) [ ZROBIONE ] */
/* Zaimplementuj Sito Erastotenesa ( algorytm do znajdowania liczb pierwszych )*/



class MinHeap final {
private:
	std::vector<int> heap;
	
	void HeapUp(int n) {

		int v = heap[n];
		int parent = (n - 1) / 2;

		while (heap[parent] > v) {
			heap[n] = heap[parent];
			n = parent;
			parent = (n - 1) / 2;

			if (n == 0) break;
		}

		heap[n] = v;

	}

	void HeapDown(int n) {
		int j;
		int v = heap[n];

		while (n <= (heap.size() - 1) / 2) {
			j = 2 * n + 1;

			if (j + 1 < heap.size()) {
				if (heap[j] > heap[j + 1]) j++;
			}

			if (j < heap.size() && heap[j] < v) {
				heap[n] = heap[j];
				n = j;
			}
			else break;
		}

		heap[n] = v;
	}

public:
	MinHeap() {};

	void Insert(int n) {
		heap.push_back(n);
		this->HeapUp(heap.size() - 1);
	}

	void DeleteMin() {
		heap[0] = heap[heap.size() - 1];
		heap.pop_back();
		this->HeapDown(0);
	}

	void PrintHeap() {
		for (auto it = heap.begin(); it != heap.end(); ++it) {
			std::cout << *it << " ";
		}
	}
};

int Partition(std::vector<int>& vec, int left, int right) {
	int i = left;
	int j = right;
	int v = vec[left];

	while (i < j) {
		while (vec[i] < v) i++;
		while (vec[j] > v) j--;

		if (i < j) {
			std::swap(vec[i], vec[j]);
		}
	}
	
	return j;
}

void QuickSort(std::vector<int>& vec, int left, int right) {
	
	if (left < right) {
		short j = Partition(vec, left, right);
		QuickSort(vec, left, j - 1);
		QuickSort(vec, j + 1, right);
	}
}

void IterativeQuickSort(std::vector<int>& vec, int left, int right) {
	std::stack<std::pair<int, int>> s;
	s.push(std::pair<int, int>(left, right));

	while (s.size() != 0) {
		auto ind = s.top();
		s.pop();

		int l = ind.first;
		int r = ind.second;

		while (l < r) {
			short j = Partition(vec, l, r);
			s.push(std::pair<int, int>(j + 1, r));
			r = j - 1;
		}
	}
}

/* Time: O(n^2) Space: O(1)*/
void BubbleSortOptmizied(std::vector<int>& vec) {
	int i, j;
	bool swapped = true;

	for (i = 0; i < vec.size() - 1; i++) {
		swapped = false;

		for (j = 0; j < vec.size() - i - 1; j++) {
			if (vec[j] > vec[j + 1]) {
				std::swap(vec[j], vec[j + 1]);
				swapped = true;
			}
		}

		if (!swapped) break;
	}
}

void PrintVector(std::vector<int>& vec) {
	for (auto it = vec.begin(); it != vec.end(); ++it) {
		std::cout << *it << " ";
	}

	std::cout << "\n";
}

void CountSort(std::vector<int>& vec, int exponential) {
	std::vector<int> cnt, out;

	cnt.resize(10);
	out.resize(vec.size());

	int i, j;

	for (j = 0; j < cnt.size(); j++) cnt[j] = 0;
	for (i = 0; i < vec.size(); i++) cnt[(vec[i] / exponential) % 10]++;
	for (j = 1; j < cnt.size(); j++) cnt[j] += cnt[j - 1];
	for (i = vec.size() - 1; i >= 0; i--) {
		out[cnt[(vec[i] / exponential) % 10] - 1] = vec[i];
		cnt[(vec[i] / exponential) % 10]--;
	}

	for (i = 0; i < vec.size(); i++) vec[i] = out[i];
}

void RadixSort(std::vector<int>& vec) {
	int exp;
	int i;
	int max = vec[0];

	for (i = 1; i < vec.size(); i++) if (vec[i] > max) max = vec[i];

	for (exp = 1; max / exp > 0; exp *= 10) {
		CountSort(vec, exp);
	}
}

/* Time: O(n^2)*/
void SelectionSort(std::vector<int>& vec) {
	int i, j, min;

	for (i = 0; i < vec.size(); i++) {
		min = i;

		for (j = i + 1; j < vec.size(); j++) {
			if (vec[j] < vec[min]) min = j;
		}

		if (min != i) std::swap(vec[i], vec[min]);
	}
}

// Time: O(n^2)
void InsertionSort(std::vector<int>& vec) {
	int i, j, v;

	for (i = 1; i < vec.size(); i++) {
		j = i;
		v = vec[i];

		while (j > 0 && vec[j - 1] > v) {
			vec[j] = vec[j - 1];
			j--;
		}
		
		if (j != i) vec[j] = v;
	}
}

/* Time: O(log(n)) */
void HeapDown(std::vector<int>& vec, int size ,int n) {
	int j;
	int v = vec[n];
	
	while (n <= (size - 1) / 2) {
		j = 2 * n + 1;

		if (j + 1 < size){
			if (vec[j + 1] > vec[j]) j++;
		}

		if (j < size && vec[j] > v) {
			vec[n] = vec[j];
			n = j;
		}
		else break;
	}

	vec[n] = v;
}

/* Time: O(n) */
void Construct(std::vector<int>& vec) {
	int i;

	for (i = (vec.size() - 1) / 2; i >= 0; i--) HeapDown(vec, vec.size(), i);
}

void HeapSort(std::vector<int>& vec) {
	int i;
	int n = vec.size();
	Construct(vec);

	for (i = n - 1; i > 0; i--) {
		std::swap(vec[0], vec[i]);
		HeapDown(vec, --n, 0);
	}
}

int HoareAlgorithm(std::vector<int>& vec, int left, int right, int k) {

	if (left < right) {
		short j = Partition(vec, left, right);
		if (k - 1 == right - j) return vec[j];
		else if (k - 1 < right - j) return HoareAlgorithm(vec, j + 1, right, k);
		else return HoareAlgorithm(vec, left, j - 1, k - (right - j + 1));
	}
	else return vec[left];

}

int IterativeHoareAlgorithm(std::vector<int>& vec, int left, int right, int k) {
	std::stack<std::pair<int, int>> s;
	s.push(std::pair<int, int>(left, right));

	while (s.size() != 0) {
		auto ind = s.top();
		s.pop();

		int l = ind.first;
		int r = ind.second;

		if (l < r) {
			short j = Partition(vec, l, r);
			if (k - 1 == r - j) return vec[j];
			else if (k - 1 < r - j) s.push(std::pair<int, int>(j + 1, r));
			else {
				s.push(std::pair<int, int>(l, j - 1));
				k -= (right - j + 1);
			}
		}
	}
}

void PartialInsertionSort(std::vector<int>& vec, int left, int right) {
	int i, j, v;

	for (i = left + 1; i < right; i++) {
		j = i;
		v = vec[i];

		while (j > left && vec[j - 1] > v) {
			vec[j] = vec[j - 1];
			j--;
		}
		
		if (j != i) vec[j] = v;
	}
}

int FindMedian(std::vector<int>& vec, int left, int n) {
	PartialInsertionSort(vec, left, left + n);
	return vec[left + n / 2];
}

int Partition(std::vector<int>& vec, int left, int right, int pivot)
{
	int i, j, v;

	for (i = left; i < right; i++) {
		if (vec[i] == pivot) break;
	}

	std::swap(vec[right], vec[i]);

	j = left;
	for (i = left; i < right; i++) {
		if (vec[i] <= pivot) {
			if (j != i) {
				std::swap(vec[i], vec[j]);
			}
			j++;
		}
	}

	std::swap(vec[j], vec[right]);
	return j;
}

int BlumFloydPrattRivestTarjanAlgorithm(std::vector<int>& vec, int left, int right, int k) {
	
	while (k > 0 && k <= (right - left) + 1) {
		int n = (right - left) + 1;

		std::vector<int> medians;
		medians.resize((n + 4) / 5);
		int i;

		for (i = 0; i < n / 5; i++) {
			medians[i] = FindMedian(vec, left + i * 5, 5);
		}

		if (i * 5 < n) {
			medians[i] = FindMedian(vec, left + i * 5, n % 5);
			i++;
		}

		int median = (i == 1) ? medians[i - 1] : BlumFloydPrattRivestTarjanAlgorithm(medians, 0, i - 1, i / 2);
		int j = Partition(vec, left, right, median);
		

		if (k - 1 == j - left) return vec[j];
		else if (k - 1 < j - left) return BlumFloydPrattRivestTarjanAlgorithm(vec, left, j - 1, k);
		else return BlumFloydPrattRivestTarjanAlgorithm(vec, j + 1, right, k - j + left - 1);
	}
}

int BinarySearach(std::vector<int>& vec, int left, int right, int k) {
	if (left > right) return -1;

	int mid = left + ((right - left) / 2);

	if (vec[mid] == k) return mid;
	else if (vec[mid] < k) return BinarySearach(vec, mid + 1, right, k);
	else return BinarySearach(vec, left, mid - 1, k);
}

int IterativeBinarySearch(std::vector<int>& vec, int left, int right, int k) {
	while (left <= right) {
		int mid = left + ((right - left) / 2);

		if (vec[mid] == k)  return mid;
		else if (vec[mid] < k) left = mid + 1;
		else right = mid - 1;
	}
}

void BorderQuickSort(std::vector<int>& vec, int left, int right) {
	
	if (left >= right) return;

	int i = left;
	int b = left - 1;
	int v = vec[right];

	while (i < right) {
		if (vec[i] < v) {
			b++;
			if (i != b) std::swap(vec[b], vec[i]);
		}

		i++;
	}

	b++;
	std::swap(vec[b], vec[right]);

	BorderQuickSort(vec, left, b - 1);
	BorderQuickSort(vec, b + 1, right);
}

void Merge(std::vector<int>& vec, int left, int mid, int right) {
	int i, j, k;

	std::vector<int> nArr, mArr;
	int n = mid - left + 1;
	int m = right - mid;

	nArr.resize(n);
	mArr.resize(m);

	for (i = 0; i < n; i++) {
		nArr[i] = vec[left + i];
	}

	for (j = 0; j < m; j++) {
		mArr[j] = vec[mid + j + 1];
	}

	i = 0;
	j = 0;
	k = left;

	while (i < n && j < m) {
		if (nArr[i] < mArr[j]) {
			vec[k] = nArr[i];
			i++;
		}
		else {
			vec[k] = mArr[j];
			j++;
		}
		k++;
	}

	while (i < n) {
		vec[k] = nArr[i];
		i++;
		k++;
	}

	while (j < m) {
		vec[k] = mArr[j];
		j++;
		k++;
	}
}

void MergeSort(std::vector<int>& vec, int left, int right) {
	if (left < right) {
		int mid = left + ((right - left) / 2);
		
		MergeSort(vec, left, mid);
		MergeSort(vec, mid + 1, right);

		Merge(vec, left, mid, right);
	}
}

void ShellSort(std::vector<int>& vec) {
	int gap, i;

	for (gap = vec.size() / 2; gap > 0; gap /= 2) {
		for (i = gap; i < vec.size(); i++) {
			int j = i;
			int v = vec[i];

			while (j >= gap && vec[j - gap] > v) {
				vec[j] = vec[j - gap];
				j -= gap;
			}

			if (j != i) vec[j] = v;
		}
	}
}

int EuklidesAlgorithm(int a, int b) {
	if (b == 0) return a;
	return EuklidesAlgorithm(b, a % b);
}

int IterativeEuklidesAlgorithm(int a, int b) {
	int c;

	while (b != 0) {
		c = a % b;
		a = b;
		b = c;
	}

	return a;
}

void SitoErastotenesa(int n) {
	std::vector<bool> primes;
	primes.resize(n + 1);
	int i, j;

	for (i = 2; i <= n; i++) primes[i] = true;
	for (i = 2; i <= std::sqrt(n); i++) {
		if (primes[i]) {
			for (j = 2 * i; j <= n; j += i) primes[j] = false;
		}
	}

	for (i = 2; i <= n; i++) {
		if (primes[i]) std::cout << i << "\t";
	}

}

int main() {
	std::vector<int> v = { 5,8,3,9,0,7,44,55,2,43,11, 22 };
	
	SitoErastotenesa(113);
}