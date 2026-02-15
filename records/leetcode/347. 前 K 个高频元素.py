# Author: https://leetcode.cn/u/233-zu/

from collections import Counter

class SolutionA:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        return [key for key, value in Counter(nums).most_common(k)]


class SolutionB:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        import heapq
        from collections import Counter

        count = Counter(nums)

        heap = []
        for num, freq in count.items():
            heapq.heappush(heap, (freq, num))

            if len(heap) > k:
                heapq.heappop(heap)

        return [num for freq, num in heap]