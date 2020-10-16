#include <iostream>
#include <vector>
#include <stack>
#include <queue>
#include <algorithm>
#include <cmath>

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
/* Zaimplementuj ShellSort ( dowolny algorytm licz¹cy dystans ) [ ZROBIONE ] */
/* Zaimplementuj Algorytm Euklidesa ( rekurencyjnie oraz iteracyjnie ) [ ZROBIONE ] */
/* Zaimplementuj Sito Erastotenesa ( algorytm do znajdowania liczb pierwszych ) [ ZROBIONE ] */
/* Zaimplementuj Wyszukiwanie interpolacyjne ( Rozk³ad Bernoulliego ) [ ZROBIONE ] */
/* Zaimplementuj Drzewo poszukiwañ binarnych BST ( search, insert, remove, Depth / Breadth First Search ) [ ZROBIONE ]*/
/* Zaimplementuj Drzewo AVL ( search, insert, remove, Depth / Breadth First Search ) [ ZROBIONE ] */
/* Zaimplementuj Drzewo Splay ( search, insert, remove, Depth / Breadth First Search ) [ ZROBIONE ] */
/* Przeanalizuj funkcje mieszaj¹ce wraz z metodami: metoda ³añcuchowa, metoda otwartych adresów [ ZROBIONE ] */
/* Zaimplementuj funkcje operate na manipulacji bitami ( and, or, xor, shifts, neg ) [ ZROBIONE ] */
/* Zaimplementuj Algorytm, który sprawdzi czy liczba jest palindromem [ ZROBIONE ]*/
/* Zaimplementuj Algorytm, który policzy silnie du¿ej liczby ( np n = 10e3 ) [ ZROBIONE ] */

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
void HeapDown(std::vector<int>& vec, int size, int k) {
	int v = vec[k];
	int j;

	while (k <= (size - 1) / 2) {
		j = 2 * k + 1;

		if (j + 1 < size) {
			if (vec[j] < vec[j + 1]) j++;
		}

		if (j < size && vec[j] > v) {
			vec[k] = vec[j];
			k = j;
		}
		else break;
	}

	vec[k] = v;
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
			if (b != i) std::swap(vec[i], vec[b]);
		}

		i++;
	}

	b++;
	std::swap(vec[b], vec[right]);

	BorderQuickSort(vec, left, b - 1);
	BorderQuickSort(vec, b + 1, right);
}

void Merge(std::vector<int>& vec, int left, int mid, int right) {

	std::vector<int> nArr, mArr;
	int n = mid - left + 1;
	int m = right - mid;

	nArr.resize(n);
	mArr.resize(m);

	int i;

	for (i = 0; i < n; i++) {
		nArr[i] = vec[left + i];
	}

	for (i = 0; i < m; i++) {
		mArr[i] = vec[mid + 1 + i];
	}

	i = 0;
	int j = 0;
	int k = left;

	while (i < n && j < m) {
		if (nArr[i] > mArr[j]) {
			vec[k] = mArr[j];
			j++;
		}
		else {
			vec[k] = nArr[i];
			i++;
		}

		k++;
	}

	while (i < n) {
		vec[k] = nArr[i];
		k++;
		i++;
	}

	while (j < m) {
		vec[k] = mArr[j];
		k++;
		j++;
	}
}

void MergeSort(std::vector<int>& vec, int left, int right) {
	if (left < right) {
		int mid = left + ((right - left)) / 2;

		MergeSort(vec, left, mid);
		MergeSort(vec, mid + 1, right);

		Merge(vec, left, mid, right);
	}
}

void ShellSort(std::vector<int>& vec) {
	int gap, i, j, v;

	for (gap = vec.size() / 2; gap > 0; gap /= 2) {
		for (i = gap; i < vec.size(); i++) {
			j = i;
			v = vec[i];

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
	while (b != 0) {
		int c = a % b;
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

/* Wyszukiwanie interpolacyjne z heurystykami dzia³a w czasie O(log log n ) */
/* Heurystyka to metoda znajdowania rozwi¹zañ, dla której nie ma gwarancji znalezienia rozwi¹zania 
optymalnego lub nawet prawid³owego, u¿ywamy go pe³ny algorytm jest zbyt kosztowny lub nawet nieznany. */
/* Metoda ta s³u¿y do znalezienia rozwi¹zañ przybli¿onych, na podstawie których wyliczamy ostateczny rezultat 
pe³nego algorytmu */
/* Stosujemy przy równomiernym rozk³adzie zbioru S */

int InterpolationSearch(std::vector<int>& vec, int left, int right, int k) {
	while (left < right && vec[left] <= k && k <= vec[right]) {
		int idx = left + (((double)(right - left) / (vec[right] - vec[left])) * (k - vec[left]));

		if (vec[idx] == k) return idx;
		else if (vec[idx] < k) left = idx + 1;
		else right = idx - 1;
	}
}

/* BST */

struct Node {
	int value;
	Node* left, * right;

	Node(int value, Node* left, Node* right) : value(value), left(left), right(right) {};
};

class BinarySearchTree {
private:
	Node* root = nullptr;
public:
	bool Search(int value) {
		return this->Search(this->root, value);
	}

	void Insert(int value) {
		if (this->Search(value)) return;
		this->root = this->Insert(this->root, value);
	}

	void Remove(int value) {
		if (!this->Search(value)) return;
		this->Remove(this->root, value);
	}

	void PreOrder() {
		this->PreOrder(this->root);
	}

	void InOrder() {
		this->InOrder(this->root);
	}

	void PostOrder() {
		this->PostOrder(this->root);
	}
	void LevelOrder() {
		this->LevelOrder(this->root);
	}


private:

	bool Search(Node* node, int value) {
		if (node == nullptr) return false;

		if (node->value < value) return this->Search(node->right, value);
		else if (node->value > value) return this->Search(node->left, value);
		else return true;
	}

	Node* Insert(Node* node, int value) {
		if (node == nullptr) node = new Node(value, nullptr, nullptr);
		
		if (node->value < value) node->right = this->Insert(node->right, value);
		else if (node->value > value) node->left = this->Insert(node->left, value);
		
		return node;
	}

	Node* Remove(Node* node, int value) {
		if (node == nullptr) return nullptr;
		else if (node->value < value) node->right = this->Remove(node->right, value);
		else if (node->value > value) node->left = this->Remove(node->left, value);
		else {
			if (node->left == nullptr) {
				Node* temp = node->right;
				delete node;
				return temp;
			}
			else if (node->right == nullptr) {
				Node* temp = node->left;
				delete node;
				return temp;
			}
			else {
				Node* temp = this->FindMin(node->right);
				node->value = temp->value;
				node->right = this->Remove(node->right, temp->value);
			}
		}

		return node;
	}

	Node* FindMin(Node* node) {
		while (node->left != nullptr) {
			node = node->left;
		}

		return node;
	}

	void PreOrder(Node* node) {
		if (node == nullptr) return;
		std::cout << node->value << "\t";
		this->PreOrder(node->left);
		this->PreOrder(node->right);
	}

	void InOrder(Node* node) {
		if (node == nullptr) return;
		this->InOrder(node->left);
		std::cout << node->value << "\t";
		this->InOrder(node->right);
	}

	void PostOrder(Node* node) {
		if (node == nullptr) return;
		this->PostOrder(node->left);
		this->PostOrder(node->right);
		std::cout << node->value << "\t";
	}

	void LevelOrder(Node* node) {
		if (node == nullptr) return;
		std::queue<Node*> que;
		que.push(node);

		while (que.size() != 0) {
			Node* front = que.front();
			que.pop();

			std::cout << front->value << "\t";
			if (front->left != nullptr) que.push(front->left);
			if (front->right != nullptr) que.push(front->right);

		}
	}
};

int GetMax(int a, int b) {
	return a > b ? a : b;
}

struct AVLNode {
	AVLNode* left, * right;
	int value, bf, height;

	AVLNode(int value, AVLNode* left, AVLNode* right) : value(value), left(left), right(right), height(0) {};
};

class AVLTree {
private:
	AVLNode* root = nullptr;
public:

	bool Search(int value) {
		return this->Search(this->root, value);
	}

	void Insert(int value) {
		if (this->Search(value)) return;
		this->root = this->Insert(this->root, value);
	}

	void Remove(int value) {
		if (!this->Search(value)) return;
		this->root = this->Remove(this->root, value);
	}

	void PreOrder() {
		this->PreOrder(this->root);
	}

	void InOrder() {
		this->InOrder(this->root);
	}

	void PostOrder() {
		this->PostOrder(this->root);
	}

	void LevelOrder() {
		this->LevelOrder(this->root);
	}

private:

	bool Search(AVLNode* node, int value) {
		if (node == nullptr) return false;

		if (node->value < value) return this->Search(node->right, value);
		else if (node->value > value) return this->Search(node->left, value);
		else return true;
	}

	AVLNode* Insert(AVLNode* node, int value) {
		if (node == nullptr) node = new AVLNode(value, nullptr, nullptr);
		
		if (node->value < value) node->right = this->Insert(node->right, value);
		else if (node->value > value) node->left = this->Insert(node->left, value);

		this->UpdateBalanceFactor(node);
		return this->ReBalanceTree(node);
	}

	AVLNode* Remove(AVLNode* node, int value) {
		if (node == nullptr) return nullptr;

		if (node->value < value) node->right = this->Remove(node->right, value);
		else if (node->value > value) node->left = this->Remove(node->left, value);	
		else
		{
			if (node->left == nullptr) {
				AVLNode* temp = node->right;
				delete node;
				return temp;
			}
			else if (node->right == nullptr) {
				AVLNode* temp = node->left;
				delete node;
				return temp;
			}
			else {
				AVLNode* temp = this->FindMin(node->right);
				node->value = temp->value;
				node->right = this->Remove(node->right, temp->value);
			}
		}

		this->UpdateBalanceFactor(node);
		this->ReBalanceTree(node);
		return node;
	}

	AVLNode* FindMin(AVLNode* node) {
		while (node->left != nullptr) {
			node = node->left;
		}
		return node;
	}

	void UpdateBalanceFactor(AVLNode* node) {
		const int leftHeight = node->left == nullptr ? -1 : node->left->height;
		const int rightHeight = node->right == nullptr ? -1 : node->right->height;

		node->height = GetMax(leftHeight, rightHeight) + 1;
		node->bf = leftHeight - rightHeight;
	}

	AVLNode* ReBalanceTree(AVLNode* node) {
		if (node->bf > 1) {
			if (node->left->bf >= 0) {
				return this->RightRotation(node);
			}
			else {
				return this->LeftRightRotation(node);
			}
		} 
		else if (node->bf < -1) {
			if (node->right->bf <= 0) {
				return this->LeftRotation(node);
			}
			else {
				return this->RightLeftRotation(node);
			}
		}
		return node;
	}

	AVLNode* LeftRotation(AVLNode* node) {
		return this->RotateLeft(node);
	}

	AVLNode* RightRotation(AVLNode* node) {
		return this->RotateRight(node);
	}

	AVLNode* LeftRightRotation(AVLNode* node) {
		node->left = this->LeftRotation(node->left);
		return this->RightRotation(node);
	}

	AVLNode* RightLeftRotation(AVLNode* node) {
		node->right = this->RightRotation(node->right);
		return this->LeftRotation(node);
	}

	AVLNode* RotateLeft(AVLNode* node) {
		AVLNode* parent = node->right;
		node->right = parent->left;
		parent->left = node;

		this->UpdateBalanceFactor(node);
		this->UpdateBalanceFactor(parent);

		return parent;
	}

	AVLNode* RotateRight(AVLNode* node) {
		AVLNode* parent = node->left;
		node->left = parent->right;
		parent->right = node;

		this->UpdateBalanceFactor(node);
		this->UpdateBalanceFactor(parent);

		return parent;
	}

	void PreOrder(AVLNode* node) {
		if (node == nullptr) return;

		std::cout << node->value << "\t";
		this->PreOrder(node->left);
		this->PreOrder(node->right);
	}

	void InOrder(AVLNode* node) {
		if (node == nullptr) return;

		this->InOrder(node->left);
		std::cout << node->value << "\t";
		this->InOrder(node->right);
	}

	void PostOrder(AVLNode* node) {
		if (node == nullptr) return;

		this->PostOrder(node->left);
		this->PostOrder(node->right);
		std::cout << node->value << "\t";
	}

	void LevelOrder(AVLNode* node) {
		std::queue<AVLNode*> nodes;
		nodes.push(node);

		while (nodes.size() != 0) {
			AVLNode* first = nodes.front();
	
			std::cout << first->value << "\t";

			if (first->left != nullptr) nodes.push(first->left);
			if (first->right != nullptr) nodes.push(first->right);

			nodes.pop();
		}
	}
};

class SplayTree {
private:
	Node* root = nullptr;
public:
	void Insert(int value) {
		this->root = this->Insert(this->root, value);
	}

	void Remove(int value) {
		this->root = this->Remove(this->root, value);
	}

	void PreOrder() {
		this->PreOrder(this->root);
	}

	void InOrder() {
		this->InOrder(this->root);
	}

	void PostOrder() {
		this->PostOrder(this->root);
	}

	void LevelOrder() {
		this->LevelOrder(this->root);
	}

private:

	Node* Splay(Node* node, int value) {
		if (node == nullptr || node->value == value) return node;

		if (node->value > value) {

			if (node->left == nullptr) return node;

			//* Zig Zig case:
			if (node->left->value > value) {
				node->left->left = this->Splay(node->left->left, value);
				node = this->RightRotate(node);
			}

			//* Zag Zig case:
			else if (node->left->value < value) {
				node->left->right = this->Splay(node->left->right, value);

				if (node->left->right != nullptr) {
					node->left = this->LeftRotate(node->left);
				}
			}

			//* Zig case:
			return node->left == nullptr ? node : this->RightRotate(node);
		}
		else {

			if (node->right == nullptr) return node;

			//* Zig Zag case:
			if (node->right->value > value) {
				node->right->left = this->Splay(node->right->left, value);

				if (node->right->left != nullptr) {
					node->right = this->RightRotate(node->right);
				}
			}

			//* Zag Zag case:
			else if (node->right->value < value) {
				node->right->right = this->Splay(node->right->right, value);
				node = this->LeftRotate(node);
			}

			return node->right == nullptr ? node : this->LeftRotate(node);
		}
	}

	Node* Insert(Node* node, int value) {
		if (node == nullptr) node = new Node(value, nullptr, nullptr);

		if (node->value < value) node->right = this->Insert(node->right, value);
		else if (node->value > value) node->left = this->Insert(node->left, value);
		node = this->Splay(node, value);

		return node;
	}

	Node* Remove(Node* node, int value) {
		if (node == nullptr) return node;

		node = this->Splay(node, value);

		if (node->value < value) node->right = this->Remove(node->right, value);
		else if (node->value > value) node->left = this->Remove(node->left, value);
		else {

			if (node->left == nullptr) {
				Node* temp = node->right;
				delete node;
				return temp;
			}

			else if (node->right == nullptr) {
				Node* temp = node->left;
				delete node;
				return temp;
			}

			else {
				Node* temp = this->FindMin(node->right);
				node->value = temp->value;
				node->right = this->Remove(node->right, temp->value);
			}
		}
	}

	Node* FindMin(Node* node) {
		while (node->left != nullptr) {
			node = node->left;
		}

		return node;
	}

	Node* RightRotate(Node* node) {
		Node* parent = node->left;
		node->left = parent->right;
		parent->right = node;

		return parent;
	}

	Node* LeftRotate(Node* node) {
		Node* parent = node->right;
		node->right = parent->left;
		parent->left = node;

		return parent;
	}

	void PreOrder(Node* node) {
		if (node == nullptr) return;

		std::cout << node->value << "\t";
		this->PreOrder(node->left);
		this->PreOrder(node->right);
	}

	void InOrder(Node* node) {
		if (node == nullptr) return;

		this->InOrder(node->left);
		std::cout << node->value << "\t";
		this->InOrder(node->right);
	}

	void PostOrder(Node* node) {
		if (node == nullptr) return;

		this->PostOrder(node->left);
		this->PostOrder(node->right);
		std::cout << node->value << "\t";
	}

	void LevelOrder(Node* node) {
		std::queue<Node*> nodes;
		nodes.push(node);

		while (nodes.size() != 0) {
			Node* first = nodes.front();

			std::cout << first->value << "\t";

			if (first->left != nullptr) nodes.push(first->left);
			if (first->right != nullptr) nodes.push(first->right);

			nodes.pop();
		}
	}
};

short AND(int a, int b) {
	return a & b;
}

short OR(int a, int b) {
	return a | b;
}

short XOR(int a, int b) {
	return a ^ b;

}

short NEG(int a) {
	return ~a;
}

short SHIFT_LEFT(int a) {
	return a << 1;
}

short SHIFT_RIGHT(int a) {
	return a >> 1;
}

bool IsPalindrome(int v) {
	int t = v;
	int res = 0;
	int pop;

	while (t != 0) {
		pop = t % 10;
		res = res * 10 + pop;
		t /= 10;
	}

	return res == v ? true : false;
}

int Multiply(int x, int res[], int res_size) {
	int rest = 0;

	for (int i = 0; i < res_size; i++) {
		int num = res[i] * x + rest;
		res[i] = num % 10;
		rest = num / 10;
	}

	while (rest) {
		res[res_size] = rest;
		rest /= 10;
		res_size++;
	}

	return res_size;
}

void Factorial(int n) {
	int res[100000];
	res[0] = 1;
	int res_size = 1;

	for (int i = 2; i <= n; i++) {
		res_size = Multiply(i, res, res_size);
	}

	for (int i = res_size - 1; i >= 0; i--) {
		std::cout << res[i] << "";
	}

	std::cout << "\n";
}

int main() {
	Factorial(10e3);
}