#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <algorithm>
#include <cassert>
#include <utility>
#include <chrono>
#include <cstdarg>
#include <future>
#include <execution>

#define DBG_RANDOM_SLEEP_MS(a, b) {                        \
    std::this_thread::sleep_for(                           \
        std::chrono::milliseconds(abs(RandGen::num(a, b))) \
    );                                                     \
}                                                          \

class RandGen {
public:
    static std::vector<int> genVector(size_t n) {
        std::uniform_int_distribution<> dist(-1000, 1000);

        std::vector<int> v(n);
        std::for_each(v.begin(), v.end(), [&dist](int& el){
            el = dist(_e);
        });
        return v;
    }

    inline static uint64_t num(int a, int b) {
        std::uniform_int_distribution<> dist(a, b);
        return dist(_e);
    }

private:
    static std::random_device _rd;
    static std::mt19937 _e;
};

std::random_device RandGen::_rd;
std::mt19937 RandGen::_e(_rd());

inline std::pair<int, int> upperPowerOf2(int n) {
    int power = 0;
    int d = 1;
    while (d < n) {++power; d *= 2;};
    return std::make_pair(d, power);
}

[[maybe_unused]] inline void extendVectorToLeftToUpperPowerOf2(std::vector<int>& v) {
    v.insert(v.begin(), upperPowerOf2(v.size()).first - v.size(), 0);
}

std::vector<int> prefLinier(const std::vector<int>& v) {
    std::vector<int> pref(v.size());
    for (int i=0; i<pref.size(); ++i) {
        pref[i] = ((i == 0) ? 0 : pref[i - 1]) + v[i];
    }
    return pref;
}

/**
 * Naive
 * time: O(log(n))
 * work: O(n*log(n))
 */
std::vector<int> prefParallel1(const std::vector<int>& v) {
    std::vector<int> pref(v);
    std::vector<int> pref_tmp(pref.size());

    for (int d=1; d<v.size(); d*=2) {
        {
        std::vector<std::jthread> workers;
        for (int i=0; i<v.size(); ++i) 
            workers.emplace_back([&pref, &pref_tmp, i, d]{
                // DBG_RANDOM_SLEEP_MS(1000, 2000);
                pref_tmp[i] = pref[i] + ((i >= d) ? pref[i - d] : 0);
            });
        } // workers sync

        std::swap(pref, pref_tmp);
    }
    return pref;
}

/**
 * Work efficient
 * time: O(log(n)
 * work: O(n)
 *
 * ref: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda | https://www.youtube.com/watch?v=xCyy-1im7f8
 * ps: overhead in implementation, but it's done on purpose
 */
std::vector<int> prefParallel2(const std::vector<int>& v) {
    std::vector<int> pref(v.size());
    
    int block_size = upperPowerOf2(v.size()).second;
    int block_count = (v.size() + block_size - 1) / block_size;

    std::vector<std::future<std::pair<std::vector<int>, int>>> futures;

    {
    std::vector<std::jthread> calculating_workers;
    for (int block_idx=0; block_idx<block_count; ++block_idx) {
        std::packaged_task<std::pair<std::vector<int>, int>()> task([&v, block_idx, block_size]{
            int l_bound = block_idx*block_size;
            int r_bound = std::min(v.size(), static_cast<size_t>((block_idx+1)*block_size));
            // DBG_RANDOM_SLEEP_MS(1000, 2000);
            return std::make_pair(
                prefParallel1(std::vector<int>(v.begin() + l_bound, v.begin() + r_bound)),
                block_idx
            );
        });
        futures.emplace_back(task.get_future());
        calculating_workers.emplace_back(std::move(task));
    }
    
    {
    std::vector<std::jthread> moving_workers;
    for (auto& f : futures) {
        moving_workers.emplace_back([&f, &pref, block_size] {
            auto [pref_block, block_idx] = f.get();
            std::move(std::execution::par, 
                pref_block.begin(), pref_block.end(), pref.begin() + block_idx*block_size
            );
        });
    }
    } // sync pref_block moving workes

    } // sync pref_block calculating workers

    std::vector<int> pref_tops;
    for (int block_idx=0; block_idx<block_count; ++block_idx)
        pref_tops.emplace_back(pref[std::min(
            pref.size(),
            static_cast<size_t>((block_idx+1)*block_size)
        ) - 1]);
    pref_tops = prefParallel1(pref_tops);

    {
    std::vector<std::future<void>> futures;
    for (int block_idx=1; block_idx<block_count; ++block_idx) // skip first block
        futures.emplace_back(std::async(std::launch::async, [&pref, val_to_add = pref_tops[block_idx - 1], block_idx, block_size]{
     
            int l_bound = block_idx*block_size;
            int r_bound = std::min(pref.size(), static_cast<size_t>((block_idx+1)*block_size));

            for (int i=l_bound; i<r_bound; ++i)
                pref[i] += val_to_add;
        }));
    
    for (const auto& f : futures)
        f.wait();
    } // sync futures
    
    return pref;
}

int main() {
    std::vector<int> v = RandGen::genVector(1000);

    std::vector<int> pref_liniar = prefLinier(v);
    std::vector<int> pref_parallel1 = prefParallel1(v);
    std::vector<int> pref_parallel2 = prefParallel2(v);
    
    // for (int& el : v) {
    //     std::cout << el << " ";
    // }
    // std::cout << std::endl;

    // for (int& el : pref_liniar) {
    //     std::cout << el << " ";
    // }
    // std::cout << std::endl;

    // for (int& el : pref_parallel1) {
    //    std::cout << el << " ";
    // }
    // std::cout << std::endl;

    // for (int& el : pref_parallel2) {
    //    std::cout << el << " ";
    // }
    // std::cout << std::endl;
        

    assert(pref_liniar == pref_parallel1);
    assert(pref_liniar == pref_parallel2);

    return 0;
}