#pragma once
//=================================================================================================
//  cx_math.hpp
//  #include "cx_math.hpp"
//  Remotion (C) 2022 - All Rights Reserved
//=================================================================================================

#include <cstdint>
#include <cfloat>
#include <cmath>

#include <type_traits>  // std::is_same_v, std::declval, std::enable_if
#include <limits>		// std::numeric_limits
#include <bit>			// std::bit_cast

/// fast constexpr math functions

#if defined(__clang__) && __clang_major__ <= 9 
namespace std {

template <class To, class From,
	enable_if_t<conjunction_v<bool_constant<sizeof(To) == sizeof(From)>,
	is_trivially_copyable<To>, is_trivially_copyable<From>>, int> = 0>
[[nodiscard]] constexpr To bit_cast(const From& value) noexcept { return __builtin_bit_cast(To, value); }


[[nodiscard]] constexpr bool is_constant_evaluated() noexcept { return __builtin_is_constant_evaluated(); }

} // namespace std
#endif

#if defined(__clang__)
#  define CX_CLANG_PRAGMA(UnQuotedPragma) _Pragma(#UnQuotedPragma)
#else 
#  define CX_CLANG_PRAGMA(UnQuotedPragma) 
#endif

#if defined(_MSC_VER)
#  define CX_MSVC_PRAGMA(UnQuotedPragma) _Pragma(#UnQuotedPragma)
#else 
#  define CX_MSVC_PRAGMA(UnQuotedPragma) 
#endif


#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
// https://docs.microsoft.com/en-us/cpp/preprocessor/fp-contract?view=msvc-170
// https://godbolt.org/z/GM4qs7PPM
#pragma fp_contract(on)
#endif

#if defined(__clang__) || defined(__GNUC__)
#  define CX_FORCEINLINE inline __attribute__((always_inline))
#  ifndef __forceinline
#    define __forceinline inline __attribute__((always_inline))
#  endif
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
#  define CX_FORCEINLINE __forceinline
#else
#  define CX_FORCEINLINE 
#endif

///  math constants
namespace cx {

template<typename T> constexpr T E = T(2.718281828459045235360287471352662497757247093699959574966967627724076630353555L); // e (Euler's Number)
template<typename T> constexpr T ONE_OVER_E = T(0.3678794411714423215955237701614608674458111310317678345078368016974614957448981L); // 1/e 

// Omega, Ω = LambertW(1), Ω = womega(0), Ω*exp(Ω) = 1
template<typename T> constexpr T OMEGA = T(0.5671432904097838729999686622103555497538157871865125081351310792230457930866); 

template<typename T> constexpr T LOG2E = T(1.442695040888963407359924681001892137426645954152985934135449406931109219181187L); // 1/log(2) = log2(e)
template<typename T> constexpr T LOG3E = T(0.9102392266268373936142401657361070006126360572552117447263020632952810831937972L); // 1/log(3)
template<typename T> constexpr T LOG4E = T(0.7213475204444817036799623405009460687133229770764929670677247034655546095905937L); // 1/log(4)
template<typename T> constexpr T LOG5E = T(0.6213349345596118107071993881805725841234130911147952958653233293551794504916422L); // 1/log(5)
template<typename T> constexpr T LOG6E = T(0.5581106265512472537174034379326177037763393583967291328410073575505919580629214L); // 1/log(6)
template<typename T> constexpr T LOG7E = T(0.5138983423697506930446493893701893866513391649321614254055023795777233681611104L); // 1/log(7)
template<typename T> constexpr T LOG8E = T(0.4808983469629878024533082270006307124755486513843286447118164689770364063937291L); // 1/log(8)
template<typename T> constexpr T LOG9E = T(0.4551196133134186968071200828680535003063180286276058723631510316476405415968986L); // 1/log(9)
template<typename T> constexpr T LOG10E = T(0.4342944819032518276511289189166050822943970058036665661144537831658646492088688L); // 1/log(10) = log10(e)

template<typename T> constexpr T LN2 = T(0.6931471805599453094172321214581765680755001343602552541206800094933936219696955L); // log(2)
template<typename T> constexpr T LN10 = T(2.302585092994045684017991454684364207601101488628772976033327900967572609677367L); // log(10)

template<typename T> constexpr T PHI          = T(1.6180339887498948482045868343656381177203091798057628621354486227052604628189L); // Golden Ratio (phi = φ) = (1.0+sqrt(5.0))/2.0
template<typename T> constexpr T ONE_OVER_PHI = T(0.6180339887498948482045868343656381177203091798057628621354486227052604628189L); // 1 / φ

template<typename T> constexpr T PI = T(3.141592653589793238462643383279502884197169399375105820974944592307816406286198L); // pi, π,  a bit more digits as 355/113.
template<typename T> constexpr T TWO_PI = T(6.283185307179586476925286766559005768394338798750211641949889184615632812572396L); // tau, τ = 2 * pi    'PI2'  

template<typename T> constexpr T ONE_OVER_TWO_PI = T(0.15915494309189533576888376337251436203445964574046L); // 1 / (2*pi)  'invtwopi', 'PI2_INV'
template<typename T> constexpr T PI_2 = T(1.57079632679489661923132169163975144209858469968755L); // pi/2  'halfpi', 'PI05'
template<typename T> constexpr T PI_4 = T(0.78539816339744830961566084581987572104929234984378L); // pi/4  'quarterpi'
template<typename T> constexpr T ONE_OVER_PI = T(0.31830988618379067153776752674502872406891929148091L); // 1 / pi  'PI_INV'
template<typename T> constexpr T TWO_OVER_PI = T(0.63661977236758134307553505349005744813783858296183L); // 2 / pi
template<typename T> constexpr T TWO_OVER_SQRTPI = T(1.12837916709551257389615890312154517168810125865800L); // 2 / sqrt(pi)

template<typename T> constexpr T SQRT2 = T(1.414213562373095048801688724209698078569671875376948073176679737990732478462102L); // sqrt(2)
template<typename T> constexpr T ONE_OVER_SQRT2 = T(0.7071067811865475244008443621048490392848359376884740365883398689953662392310596L); // 1 / sqrt(2)

template<typename T> constexpr T SQRT3 = T(1.732050807568877293527446341505872366942805253810380628055806979451933016908798L); // sqrt(3)
template<typename T> constexpr T ONE_OVER_SQRT3 = T(0.5773502691896257645091487805019574556476017512701268760186023264839776723029325L); // 1 / sqrt(3)

template<typename T> constexpr T SQRT5 = T(2.23606797749978969640917366873127623544061835961152572427089724541052092563782); // sqrt(5)
template<typename T> constexpr T ONE_OVER_SQRT5 = T(0.4472135954999579392818347337462552470881236719223051448541794490821041851275596); // 1 / sqrt(5)

} // namespace cx

/// Function prototypes
namespace cx {

constexpr float floor(float x) noexcept;
constexpr double floor(double x) noexcept;

constexpr float round(float d) noexcept;
constexpr double round(double d) noexcept;

/// Trigonometric functions.

constexpr float cos(float x) noexcept;
constexpr double cos(double x) noexcept;

constexpr float sin(float x) noexcept;
constexpr double sin(double x) noexcept;

constexpr float tan(float x) noexcept;
constexpr double tan(double x) noexcept;


/// Inverse trigonometric functions.

constexpr float acos(float x) noexcept;
constexpr double acos(double x) noexcept;

constexpr float asin(float x) noexcept;
constexpr double asin(double x) noexcept;

constexpr float atan(float x) noexcept;
constexpr double atan(double x) noexcept;

/// Logarithms, exponential and power functions

constexpr float sqrt(float x) noexcept;
constexpr double sqrt(double x) noexcept;

constexpr float cbrt(float x) noexcept;
constexpr double cbrt(double x) noexcept;

constexpr float pow(float x, float y) noexcept;
constexpr double pow(double x, double y) noexcept;

constexpr float log(float x) noexcept;
constexpr double log(double x) noexcept;

constexpr float exp(float x) noexcept;
constexpr double exp(double x) noexcept;

} // namespace cx



/// Fast constexpr math functions
namespace cx {

/// select(m, t, f) between two values based on a boolean condition.
template<typename T> constexpr __forceinline
T select(bool m, T a_true, T b_false) noexcept { return m ? a_true : b_false; }

// Conditional add: For all vector elements i: result[i] = m[i] ? (a[i] + b[i]) : a[i]
template<typename T> constexpr __forceinline
T if_add(bool m, T a, T b) noexcept { return select(m, a + b, a); }

// Conditional sub: For all vector elements i: result[i] = m[i] ? (a[i] - b[i]) : a[i]
template<typename T> constexpr __forceinline
T if_sub(bool m, T a, T b) noexcept { return select(m, a - b, a); }

// Conditional mul: For all vector elements i: result[i] = m[i] ? (a[i] * b[i]) : a[i]
template<typename T> constexpr __forceinline
T if_mul(bool m, T a, T b) noexcept { return select(m, a * b, a); }


constexpr bool same_value(double x, double y) noexcept { return std::bit_cast<int64_t>(x) == std::bit_cast<int64_t>(y); }
constexpr bool same_value(float x, float y) noexcept { return std::bit_cast<int32_t>(x) == std::bit_cast<int32_t>(y); }


/// abs(x) computes absolute value of an integral value |x|
constexpr float fabsf(float x) { return std::bit_cast<float>(0x7fffffffU & std::bit_cast<uint32_t>(x)); }

/// abs(x) computes absolute value of an integral value |x|
constexpr float fabs(float x) { return std::bit_cast<float>(0x7fffffffU & std::bit_cast<uint32_t>(x)); }

/// abs(x) computes absolute value of an integral value |x|
constexpr double fabs(double x) { return std::bit_cast<double>(INT64_C(0x7fffffffffffffff) & std::bit_cast<uint64_t>(x)); }

/// abs(x)  |x|
template<typename T> __forceinline constexpr
T abs(T d) { 
	if constexpr (std::is_unsigned_v<T>) { return d; }
	else { return (d >= T(0)) ? d : -d; }
}

/// Number of zeros leading the binary representation of `x`.
/// leading_zeros(1) == 31
template<typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
constexpr U leading_zeros(U x) noexcept { return std::countl_zero(x); }

/// Number of zeros trailing the binary representation of `x`.
/// trailing_zeros(2) == 1
template<typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
constexpr U trailing_zeros(U x) noexcept { return std::countr_zero(x); }

/// Number of ones leading the binary representation of `x`.
/// leading_ones(uint32_t(ipow(2,32) - 2)) == 31
template<typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
constexpr U leading_ones(U x) noexcept { return std::countl_one(x); }

/// Number of ones trailing the binary representation of `x`.
/// trailing_ones(3u) == 2
template<typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
constexpr U trailing_ones(U x) noexcept { return std::countr_one(x); }

/// Number of ones in the binary representation of `x`.
/// count_ones(111) == 6
template<typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
constexpr U count_ones(U x) noexcept { return std::popcount(x); }

/// Checks if two 32-bit floats are equal by computing their ULPs difference.
constexpr bool almost_equal_ulps(float x, float y, int32_t max_ulps_diff = 2) noexcept {
	const int32_t ix = std::bit_cast<int32_t>(x);
	const int32_t iy = std::bit_cast<int32_t>(y);
	if ((ix < 0) != (iy < 0)) { // In case the sign is different we still need to check if the floats were equal to make sure -0 is equal to +0.
		return (x == y);
	} else { return cx::abs(ix - iy) < max_ulps_diff; }
}

/// Checks if two 64-bit floats are equal by computing their ULPs difference.
constexpr bool almost_equal_ulps(double x, double y, int64_t max_ulps_diff = 2) noexcept {
	const int64_t ix = std::bit_cast<int64_t>(x);
	const int64_t iy = std::bit_cast<int64_t>(y);
	if ((ix < 0) != (iy < 0)) {	// In case the sign is different we still need to check if the floats were equal to make sure -0 is equal to +0.
		return (x == y);
	} else { return cx::abs(ix - iy) < max_ulps_diff;	}
}

/// eps(x) == max(x-prevfloat(x), nextfloat(x)-x)
__forceinline constexpr float eps(float x) noexcept { //TODO: do not work for std::numeric_limits<float>::max(), look to ulp(float)
	return cx::abs(std::bit_cast<float>(std::bit_cast<int32_t>(x) + 1) - x);
}

/// eps(x) == max(x-prevfloat(x), nextfloat(x)-x)
__forceinline constexpr double eps(double x) noexcept { //TODO: do not work for std::numeric_limits<double>::max(), look to ulp(float)
	return cx::abs(std::bit_cast<double>(std::bit_cast<int64_t>(x) + 1) - x);
}

/// sign(x)  returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x is greater than 0.0. 
__forceinline constexpr float sign(float x) noexcept { return x < 0.0f ? -1.0f : (x == 0.0f ? 0.0f : 1.0f); }

/// sign(x)  returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x is greater than 0.0. 
__forceinline constexpr double sign(double x) noexcept { return x < 0.0 ? -1.0 : (x == 0.0 ? 0.0 : 1.0); }

/// sign(x)  returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x is greater than 0.0. 
template<typename T> __forceinline constexpr
T sign(T x) noexcept { return x < T(0) ? T(-1) : (x == T(0) ? T(0) : T(1)); }

/// Composes a floating point value with the magnitude of mag and the sign of sgn. 
__forceinline constexpr float copysign(float mag, float sgn) noexcept {
	return std::bit_cast<float>((std::bit_cast<int32_t>(mag) & ~(1 << 31)) ^ (std::bit_cast<int32_t>(sgn) & (1 << 31)));
}
/// Create value with given magnitude, copying sign of second value. 
__forceinline constexpr float copysignf(float mag, float sign) { return cx::copysign(mag, sign); }

/// Composes a floating point value with the magnitude of mag and the sign of sgn. 
__forceinline constexpr double copysign(double mag, double sgn) noexcept {
	return std::bit_cast<double>((std::bit_cast<int64_t>(mag) & ~(INT64_C(1) << 63)) ^ (std::bit_cast<int64_t>(sgn) & (INT64_C(1) << 63)));
}

/// min(a,b)
template<typename T> __forceinline constexpr
T min(const T& a, const T& b) noexcept { return a < b ? a : b; }

/// min(a,b,...)
template<typename T, typename... Ts> __forceinline constexpr
T min(const T& a, const T& b, const Ts &... ts) noexcept { return min(min(a, b), ts...); }

/// max(a,b)
template<typename T> __forceinline constexpr
T max(const T& a, const T& b) noexcept { return a > b ? a : b; }

/// max(a,b,...)
template<typename T, typename... Ts> __forceinline constexpr
T max(const T& a, const T& b, const Ts &... ts) noexcept { return max(max(a, b), ts...); }

/// a && b && ...
template <typename... Ts> __forceinline constexpr
auto and_all(const Ts& ... n) noexcept { return (n && ... ); }

/// a || b || ...
template <typename... Ts> __forceinline constexpr
auto or_all(const Ts& ... n) noexcept { return (n || ... ); }


/// Sum  Σ
template <typename... Ts> __forceinline constexpr
auto sum(const Ts& ... n) noexcept { return (n + ... ); }

/// Product  Π
template <typename... Ts> __forceinline constexpr
auto product(const Ts& ... n) noexcept { return (n * ... ); }


template<typename... Ts>
constexpr auto mean(Ts... args) noexcept {
    return sum(args...) / sizeof...(args);
}

/// Greatest common (positive) divisor (or zero if all arguments are zero).
// https://en.wikipedia.org/wiki/Binary_GCD_algorithm
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>> >
constexpr T gcd(T a, T b) noexcept {
	using U = std::make_unsigned_t<T>;
	if (a == 0) return cx::abs(b);
	if (b == 0) return cx::abs(a);
	const U za = trailing_zeros(U(a));
	const U zb = trailing_zeros(U(b));
	const U k = cx::min(za, zb);
	U u = U(cx::abs(a >> za));
	U v = U(cx::abs(b >> zb));
	while (u != v) {
		if (u > v) { std::swap(u, v); }
		v -= u;
		v >>= trailing_zeros(v);
	}
	return u << k;
}

/// Greatest common (positive) divisor (or zero if all arguments are zero).
template<typename T, typename... Ts> constexpr
T gcd(const T& a, const T& b, const Ts &... ts) noexcept { return gcd(gcd(a, b), ts...); }

/// Least common (positive) multiple (or zero if any argument is zero).
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>> >
constexpr T lcm(T a, T b) noexcept {
	if ((a == 0) || (b == 0)) return 0;
	const T pa = abs(a);
	const T pb = abs(b);
	return (pa / gcd(pa, pb)) * pb;
}
/// Least common (positive) multiple (or zero if any argument is zero).
template<typename T, typename... Ts> constexpr
T lcm(const T& a, const T& b, const Ts &... ts) noexcept { return lcm(lcm(a, b), ts...); }


/// Arithmetic geometric mean
// https://mathworld.wolfram.com/Arithmetic-GeometricMean.html
template<typename T/*, typename = std::enable_if_t<std::is_integral_v<T>>*/ >
constexpr T agm(T a, T b) noexcept {
	const T epsilon = eps(a);
	const T one_over_two = T(1) / T(2);
	T a1, b1;
	while (abs(a-b) > epsilon) {
		a1 = (a + b) * one_over_two;
		b1 = sqrt(a * b);
		a = a1;
		b = b1;
	}
	return (a + b) * one_over_two;
}

/// approximation of nth fibonacci number   phi^n / sqrt(5)
constexpr float fibonacci(float n) noexcept {
	return cx::round(cx::pow(cx::PHI<float>, n) * cx::ONE_OVER_SQRT5<float>);
}

///  approximation of nth fibonacci number   phi^n / sqrt(5)
constexpr double fibonacci(double n) noexcept {
	return cx::round(cx::pow(cx::PHI<double>, n) * cx::ONE_OVER_SQRT5<float>);
}

namespace detail {

// https://www.nayuki.io/page/fast-fibonacci-algorithms
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>> >
constexpr auto _fibonacci(T n) noexcept {
	struct fib_res { T fn; T fnp1; };
	if (n == 0) {
		return fib_res{ 0, 1 };
	}
	else {
		auto [a, b] = _fibonacci(n / 2);
		const T c = a * (b * 2 - a);
		const T d = a * a + b * b;
		if (n % 2 == 0) { // is_even(n)
			return fib_res{ c, d };
		}
		else {
			return fib_res{ d, c + d };
		}
	}
}

} // namespace detail

/// nth fibonacci number for integers !
///NOTE: work correctly only up to fibonacci(93) if T is utin64_t!
template<typename T = uint64_t, typename = std::enable_if_t<std::is_integral_v<T>> >
constexpr T fibonacci(T n) noexcept { return detail::_fibonacci(n).fn; }

/// factorial of a non-negative integer n, denoted by n!
// n! = n * (n-1) * (n-2) * ... * 1    (https://en.wikipedia.org/wiki/Factorial)
// maximal n is 170
constexpr double factorial(const int n) noexcept {
	// assert(n <= 170)
	double answer = 1;
	for (int i = 1; i <= n; i++) { answer *= i; }
	return answer;
}

/// double factorial or semifactorial of a number n, denoted by n!!
// n!! = n * (n-2) * (n-4) ...    (https://en.wikipedia.org/wiki/Double_factorial)
// maximal n is 300
constexpr double double_factorial(const int n) noexcept {
	// assert(n <= 300)
	double answer = 1;
	for (int i = n; i > 0; i -= 2) { answer *= i; }
	return answer;
}

namespace detail {

constexpr auto _bit_width(uint64_t n) noexcept { return n == 0 ? 1 : 64 - std::countl_zero(n); }

// http://projecteuclid.org/euclid.rmjm/1181070157
constexpr uint64_t estimate_num_primes(uint64_t lo, uint64_t hi) noexcept {
    return 5 + uint64_t(cx::floor( hi / (cx::log(double(hi)) - 1.12) - lo / (cx::log(double(lo)) - 1.12 * (lo > 7))));
}

//TODO: use _umul128 and _udiv128 here >>>

// Computes (a + b) % m, assumes a < m, b < m.
constexpr uint64_t addmod64(uint64_t a, uint64_t b, uint64_t m) noexcept {
	if (b >= m - a) return a - m + b;
	return a + b;
}
// Computes (a*b) % m safely, considering overflow. Requires b < m;
constexpr uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) noexcept {
	if (a == 0) return b;
	if (b <= std::numeric_limits<uint64_t>::max() / a) return (a * b) % m;
	uint64_t res = 0;
	while (a != 0) {
		if (a & 1) res = addmod64(res, b, m);
		a >>= 1;
		b = addmod64(b, b, m);
	}
	return res;
}

// Compute x^p mod m
constexpr uint64_t powermod64(uint64_t b, uint64_t e, uint64_t m) noexcept {
	uint64_t r = 1;
	b %= m;
	while (e) {
		if (e % 2 == 1) r = mulmod64(r, b, m); // r = r*b % m
		b = mulmod64(b, b, m); // b = b*b % m
		e >>= 1;
	}
	return r;
}

// Compute x^p mod m
constexpr uint32_t powermod32(uint32_t a, uint32_t b, uint32_t n) noexcept {
	uint64_t d = 1, A = a;
	do {
		if (b & 1) d = (d * A) % n;
		A = (A * A) % n;
	} while (b >>= 1);
	return (uint32_t)d;
}

// a*a % n
constexpr uint32_t square_modulo32(uint32_t a, uint32_t n) noexcept {
	return (uint32_t)(((uint64_t)a * a) % n);
}

} // namespace detail

/// is_prime_slow(n)  Returns `true` if `n` is prime, and `false` otherwise. /// O(sqrt(n))
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool is_prime_slow(T num) {
	bool result = true;
	if (num <= 1) { return false; }
	else if (num == 2 || num == 3) { return true; }
	else if ((num % 2) == 0 || num % 3 == 0) { return false; }
	else {
		for (T i = 5; (i * i) <= (num); i = (i + 6)) {
			if ((num % i) == 0 || (num % (i + 2) == 0)) {
				result = false;
				break;
			}
		}
	}
	return (result);
}

constexpr bool is_prime_slow2(int64_t n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0 || n % 5 == 0) return false;
    if (n % 7 == 0 || n % 11 == 0 || n % 13 == 0) return false;
    for (int64_t i = 30; i * i <= n; i += 30) {
        if (n % (i - 13) == 0 || n % (i - 11) == 0 ||
            n % (i - 7)  == 0 || n % (i - 1)  == 0 ||
            n % (i + 1)  == 0 || n % (i + 7)  == 0 ||
            n % (i + 11) == 0 || n % (i + 13) == 0) {
            return false;
        }
    }
    return true;
}


/// is_prime(n)  Returns `true` if `n` is prime, and `false` otherwise.
// https://cp-algorithms.com/algebra/primality_tests.html
// It's also possible to do the check with only 7 bases: 2, 325, 9375, 28178, 450775, 9780504 and 1795265022. 
// However, since these numbers (except 2) are not prime, you need to check additionally if the number you are checking is equal 
// to any prime divisor of those bases: 2, 3, 5, 13, 19, 73, 193, 407521, 299210837.
template<typename T = uint64_t, typename = std::enable_if_t<std::is_integral_v<T>> >
constexpr T is_prime(T n) noexcept {
	constexpr T small_primes[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,     73, 193, 407521, 299210837 }; // 16 primes
	for (auto sp : small_primes) {
		if (n % sp == 0) { return n == sp; }
	}
	if (n < 37 * 37) { return n > 1; }
	const auto s = std::countr_zero(uint64_t(n - 1)); // trailing_zeros(uint64_t(n - 1))
	const auto d = (n - 1) >> s;

	if constexpr (sizeof(T) == 8) { // 64 bit
		constexpr T witnesses[] = { 2, 325, 9375, 28178, 450775, 9780504, 1795265022 }; // 7 witnesses from http://miller-rabin.appspot.com/
		for (const auto& a : witnesses) {
			T x = detail::powermod64(a, d, n); // x = (a^d) % n;
			if (x == 1) continue;
			auto t = s;
			while (x != n - 1) {
				if ((t -= 1) <= 0) return false;
				x = detail::mulmod64(x, x, n); // x = (x*x) % n;
				if (x == 1) return false;
			}
		}
	}
	else {
		// if n < 4,759,123,141, it is enough to test a = 2, 7, and 61;
		constexpr T witnesses[] = { 2, 7, 61 }; // 3 witnesses
		for (const auto& a : witnesses) {
			T x = detail::powermod32(a, d, n); // x = (a^d) % n;
			if (x == 1) continue;
			auto t = s;
			while (x != n - 1) {
				if ((t -= 1) <= 0) return false;
				x = detail::square_modulo32(x, n); // x = (x*x) % n;
				if (x == 1) return false;
			}
		}
	}
	return true;
}

template <typename... Ts> constexpr
auto all_primes(const Ts& ... n) noexcept { return (is_prime(n) && ... ); }

/// clamp(value, low, high)  Returns v clamped to the range [low,high]
template <typename T> __forceinline constexpr 
T clamp(const T& v, const T& low, const T& high) noexcept { return max(min(v, high), low); }

/// x^2,  x²
template<typename T> __forceinline constexpr 
T sqr(T val) noexcept { return val * val; }

/// x^3  x³
template<typename T> __forceinline constexpr 
T cube(T val) noexcept { return val * val * val; }

/// √x, isqrt(x) 
template<typename U, typename = std::enable_if_t<std::is_integral_v<U>> >
constexpr U isqrt(U n) noexcept {
	auto shift{ detail::_bit_width(n) }; // auto shift{ n == 0 ? 1 : (sizeof(U) * 8) - std::countl_zero(n) };
	shift += shift & 1; // round up to next multiple of 2
	U result = 0;
	do { // less that 32 iterations for 64 bit int
		shift -= 2;
		result <<= 1; // left-shift the result to make the next guess
		result |= 1;  // guess that the next bit is 1
		result ^= result * result > (n >> shift); // revert if guess too high
	} while (shift != 0);
	return result;
}

/// xʸ, x^y, x to the power of y
//TODO: prevent integral overflow !
template <class T0, class T1, typename = std::enable_if_t<std::is_integral_v<T1>>>
__forceinline constexpr T0 ipow(const T0& t0, const T1& t1) {
	static_assert(std::is_integral<T1>::value, "second argument must be an integer");
	T0 a = t0;
	T1 b = t1;
	bool const recip = b < 0;
	T0 r{ static_cast<T0>(1) };
	while (1) {
		if (b & 1) { r *= a; }
		b /= 2;
		if (b == 0) { break; }
		a *= a;
	}
	return recip ? 1 / r : r;
}

/// Quick test for whether an integer is a power of 2.
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>> >
__forceinline constexpr bool is_pow2(T x) noexcept {
	// The principle is that x is a power of 2 <=> x == 1<<b <=> x-1 is all 1 bits for bits < b.
	return (x & (x - 1)) == 0 && (x >= 0);
}

/// Quick test for whether an integer is a power of 2.
template <typename... Ts> constexpr
bool all_pow2(const Ts& ... n) noexcept { return (is_pow2(n) && ...); }

/// 1,3,5... are odd 
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>> >
__forceinline constexpr bool is_odd(T x) noexcept { return x % 2; }

/// 1,3,5... are odd 
template <typename... Ts> constexpr
bool all_odd(const Ts& ... n) noexcept { return (is_odd(n) && ...); }

/// 0,2,4... are even 
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>> >
__forceinline constexpr bool is_even(T x) noexcept { return !(x % 2); }

/// 0,2,4... are even 
template <typename... Ts> constexpr
bool all_even(const Ts& ... n) noexcept { return (is_even(n) && ...); }

/// saturate(x)  Returns x saturated to the range [0,1] as follows:
__forceinline constexpr float saturate(float x) noexcept { return max(0.0f, min(1.0f, x)); }

/// saturate(x)  Returns x saturated to the range [0,1] as follows:
__forceinline constexpr double saturate(double x) noexcept { return max(0.0, min(1.0, x)); }

/// saturate(x)  Returns x saturated to the range [0,1] as follows:
template <typename T> __forceinline constexpr 
T saturate(T x) noexcept { return max(T(0), min(T(1), x)); }

constexpr float radians(float a) noexcept { return a * float(PI<double> / 180.0); }
constexpr double radians(double a) noexcept { return a * PI<double> / 180.0; }

constexpr float degrees(float a) noexcept { return a * float(180.0 / PI<double>); }
constexpr double degrees(double a) noexcept { return a * 180.0 / PI<double>; }

/// step
__forceinline constexpr float  step(float edge, float x)  noexcept  { return (x >= edge) ? 1.0f : 0.0f; }
/// step
__forceinline constexpr double step(double edge, double x) noexcept { return (x >= edge) ? 1.0  : 0.0;  }

/// perform Hermite interpolation between a and b.
template <typename T> __forceinline constexpr
T smoothstep(T a, T b, T u) {
  auto t = clamp((u - a) / (b - a), T(0.0), T(1.0));
  return t * t * (T(3.0) - T(2.0) * t);
}

/// https://www.iquilezles.org/www/articles/ismoothstep/ismoothstep.htm
template <typename T> __forceinline constexpr
T inverse_smoothstep(T y ) {
    return T(0.5) - sin(asin(T(1.0)-T(2.0)*y)/T(3.0)); //TODO: !
}

template <typename T> __forceinline constexpr
T bias(T a, T bias) {
	return a / ((1 / bias - 2) * (1 - a) + 1); //TODO: !
}

template <typename T> __forceinline constexpr
T gain(T a, T gain) {
	return (a < T(0.5)) ? bias(a * 2, gain) / 2
                    : bias(a * 2 - 1, 1 - gain) / 2 + T(0.5); //TODO: !
}

/// mix(a,b, w) performs a linear interpolation between a and b using w to weight between them.
template <typename T> __forceinline constexpr
T mix(T a, T b, T w) noexcept { return a + w * (b - a); }

///  Computes the linear interpolation between a and b, if the parameter t is inside [0, 1].
__forceinline constexpr float  lerp(float a, float b, float t)  noexcept     { return (1.0f - t) * a  +  t * b; }
///  Computes the linear interpolation between a and b, if the parameter t is inside [0, 1].
__forceinline constexpr double lerp(double a, double b, double t)  noexcept  { return (1.0  - t) * a  +  t * b; }

/// inverse lerp returns a fraction t, based on a value between a and b. 
__forceinline constexpr float  inverse_lerp(float a, float b, float value)  noexcept     { return (value - a) / (b - a); }
/// inverse lerp returns a fraction t, based on a value between a and b. 
__forceinline constexpr double inverse_lerp(double a, double b, double value)  noexcept  { return (value - a) / (b - a); }

// TODO: optimize!
__forceinline constexpr float  eerp(float a, float b, float t)  noexcept   { return cx::pow( a, 1.0f - t ) * cx::pow( b, t ); }
__forceinline constexpr double eerp(double a, double b, double t) noexcept { return cx::pow( a, 1.0  - t ) * cx::pow( b, t ); }

// TODO: optimize !
__forceinline constexpr float  inverse_eerp(float a, float b, float value) noexcept    { return cx::log( a / value ) / cx::log( a / b ); }
__forceinline constexpr double inverse_eerp(double a, double b, double value) noexcept { return cx::log( a / value ) / cx::log( a / b ); }

/// 
__forceinline constexpr float remap(float iMin, float iMax, float oMin, float oMax, float value) noexcept {
	const float t = inverse_lerp(iMin, iMax, value);
	return lerp(oMin, oMax, t);
}
/// 
__forceinline constexpr double remap(double iMin, double iMax, double oMin, double oMax, double value) noexcept {
	const double t = inverse_lerp(iMin, iMax, value);
	return lerp(oMin, oMax, t);
}

///  x==0.0f 
constexpr bool is_zero(float x) noexcept { return (std::bit_cast<int32_t>(x) & 0x7FFFFFFF) == 0; }
///  x==0.0
constexpr bool is_zero(double x) noexcept { return (std::bit_cast<int64_t>(x) & 0x7FFFFFFFFFFFFFFF) == 0; }

/// all x==0.0
template <typename... Ts> constexpr
bool all_zero(const Ts& ... n) noexcept { return (is_zero(n) && ...); }

/// Determine whether argument is a NaN.  
__forceinline constexpr bool isnan(float a) noexcept {
	auto l = std::bit_cast<uint32_t>(a);
	l &= 0x7FFFFFFF;
	return l > 0x7F800000;
}
/// Determine whether argument is a NaN.  
__forceinline constexpr bool isnan(double a) noexcept {
	const auto l = std::bit_cast<uint64_t>(a);
	return (l << 1) > 0xffe0000000000000ull;
}


__forceinline constexpr bool isinf(float x) noexcept { // duplicate
	const auto l = std::bit_cast<uint32_t>(x);
	return (l << 1) == uint32_t(0xFF000000);
}
/// Determine whether argument is infinite.  
__forceinline constexpr bool isinf(double x) noexcept { // duplicate
	const auto l = std::bit_cast<uint64_t>(x);
	return (l << 1) == 0xFFE0000000000000ull;
}

__forceinline constexpr float  pow2if(int q) { return std::bit_cast<float>(((int32_t)(q + 0x7f)) << 23); }
__forceinline constexpr double pow2i(int q)  { return std::bit_cast<double>(((int64_t)(q + 0x3ff)) << 52); }

/// x * 2ᵉˣᵖ   On binary systems (where FLT_RADIX is 2), std::scalbn is equivalent to std::ldexp. 
__forceinline constexpr float scalbnf(float x, int32_t n) noexcept {
	constexpr auto x1p127 = std::bit_cast<float>(0x7f000000); // 0x1p127f === 2 ^ 127
	constexpr auto x1p_126 = std::bit_cast<float>(0x800000); // 0x1p-126f === 2 ^ -126
	constexpr auto x1p24 = std::bit_cast<float>(0x4b800000); // 0x1p24f === 2 ^ 24
	if (n > 127) {
		x *= x1p127;
		n -= 127;
		if (n > 127) {
			x *= x1p127;
			n -= 127;
			if (n > 127) { n = 127; }
		}
	}
	else if (n < -126) {
		x *= x1p_126 * x1p24;
		n += 126 - 24;
		if (n < -126) {
			x *= x1p_126 * x1p24;
			n += 126 - 24;
			if (n < -126) { n = -126; }
		}
	}
	return x * std::bit_cast<float>((uint32_t(0x7f + n)) << 23);
}

/// x * 2ᵉˣᵖ   On binary systems (where FLT_RADIX is 2), std::scalbn is equivalent to std::ldexp. 
__forceinline constexpr double scalbn(double x, int32_t n) noexcept {
	constexpr auto x1p1023 = std::bit_cast<double>(0x7fe0000000000000); // 0x1p1023 === 2 ^ 1023
	constexpr auto x1p53 = std::bit_cast<double>(0x4340000000000000); // 0x1p53 === 2 ^ 53
	constexpr auto x1p_1022 = std::bit_cast<double>(0x0010000000000000); // 0x1p-1022 === 2 ^ (-1022)
	auto y = x;
	if (n > 1023) {
		y *= x1p1023;
		n -= 1023;
		if (n > 1023) {
			y *= x1p1023;
			n -= 1023;
			if (n > 1023) {
				n = 1023;
			}
		}
	}
	else if (n < -1022) {
		/* make sure final n < -53 to avoid double rounding in the subnormal range */
		y *= x1p_1022 * x1p53;
		n += 1022 - 53;
		if (n < -1022) {
			y *= x1p_1022 * x1p53;
			n += 1022 - 53;
			if (n < -1022) {
				n = -1022;
			}
		}
	}
	return y * std::bit_cast<double>((uint64_t(0x3ff + n)) << 52);
}

#if 1
/// Multiplies a floating point value x by the number 2 raised to the n power.
__forceinline constexpr float ldexp(float x, int32_t n) {
	uint32_t ex = 0x7F800000u;
	uint32_t ix = std::bit_cast<uint32_t>(x);
	ex &= ix;              // extract old exponent;
	ix = ix & ~0x7F800000u;  // clear exponent
	n = (n << 23) + ex;
	ix |= n; // insert new exponent
	return std::bit_cast<float>(ix);
}

/// Multiplies a floating point value x by the number 2 raised to the n power.
__forceinline constexpr double ldexp(double x, int32_t n) {
	uint64_t ex = 0x7ff0000000000000;
	uint64_t ix = std::bit_cast<uint64_t>(x);
	ex &= ix;
	ix = ix & ~0x7ff0000000000000;  // clear exponent
	const int64_t n64 = (int64_t(n) << 52) + ex;
	ix |= n64; // insert new exponent
	return std::bit_cast<double>(ix);
}
#else 
/// ldexp
constexpr float ldexpf(float x, int exp) {
	if (exp > 300) exp = 300;
	if (exp < -300) exp = -300;

	int e0 = exp >> 2;
	if (exp < 0) e0++;
	if (-50 < exp && exp < 50) e0 = 0;
	int e1 = exp - (e0 << 2);

	float p = pow2if(e0);
	float ret = x * pow2if(e1) * p * p * p * p;
	return ret;
}

/// ldexp
constexpr double ldexp(double x, int exp) {
	if (exp > 2100) exp = 2100;
	if (exp < -2100) exp = -2100;

	int e0 = exp >> 2;
	if (exp < 0) e0++;
	if (-100 < exp && exp < 100) e0 = 0;
	int e1 = exp - (e0 << 2);

	double p = pow2i(e0);
	double ret = x * pow2i(e1) * p * p * p * p;
	return ret;
}
#endif

/// Return next representable double-precision floating-point value after argument x in the direction of y. 
__forceinline constexpr float nextafterf(float a, float b) {
	uint32_t ia = std::bit_cast<uint32_t>(a); // memcpy(&ia, &a, sizeof(float));
	const uint32_t ib = std::bit_cast<uint32_t>(b); // memcpy(&ib, &b, sizeof(float));
	if (isnan(a) || isnan(b)) { return a + b; } // NaN
	if (((ia | ib) << 1) == 0) { return b; }
	if (a == 0.0f) { return copysignf(1.401298464e-045f, b); }  // crossover 
	if ((a < b) && (a < 0.0f)) ia--;
	if ((a < b) && (a > 0.0f)) ia++;
	if ((a > b) && (a < 0.0f)) ia++;
	if ((a > b) && (a > 0.0f)) ia--;
	a = std::bit_cast<float>(ia); // memcpy(&a, &ia, sizeof(float));
	return a;
}

/// Return next representable double-precision floating-point value after argument x in the direction of y. 
__forceinline constexpr double nextafter(double a, double b) {
	uint64_t ia = std::bit_cast<uint64_t>(a); // memcpy(&ia, &a, sizeof(double));
	const uint64_t ib = std::bit_cast<uint64_t>(b); // memcpy(&ib, &b, sizeof(double));
	if (isnan(a) || isnan(b)) { return a + b; } // NaN 
	if (((ia | ib) << 1) == 0ULL) { return b; }
	if (a == 0.0) { return copysign(4.9406564584124654e-324, b); } // crossover 
	if ((a < b) && (a < 0.0)) ia--;
	if ((a < b) && (a > 0.0)) ia++;
	if ((a > b) && (a < 0.0)) ia++;
	if ((a > b) && (a > 0.0)) ia--;
	a = std::bit_cast<double>(ia); // memcpy(&a, &ia, sizeof(double));
	return a;
}

/// Decomposes given floating point value x into a normalized fraction and an integral power of two.
/// x = significand * 2^exponent, returns significand, pw2 is exponent.
__forceinline constexpr float frexp(float x, int32_t* pw2) {
	uint32_t ex = 0x7F800000u;              // exponent mask
	uint32_t ix = std::bit_cast<uint32_t>(x);
	ex &= ix;
	ix &= ~0x7F800000u;  // clear exponent
	*pw2 = int32_t(ex >> 23) - 126; // compute exponent
	ix |= 0x3F000000u;         // insert exponent +1 in x
	return std::bit_cast<float>(ix);
}

/// Decomposes given floating point value x into a normalized fraction and an integral power of two.
__forceinline constexpr double frexp(double x, int32_t* pw2) {
	uint64_t ex = 0x7ff0000000000000;              // exponent mask
	uint64_t ix = std::bit_cast<uint64_t>(x);
	ex &= ix;
	ix &= ~0x7ff0000000000000;  // clear exponent
	*pw2 = int32_t(ex >> 52) - 1022; // compute exponent
	ix |= 0x3fe0000000000000;         // insert exponent +1 in x
	return std::bit_cast<double>(ix);
}
#if 0
/// frexp,  Get significand and exponent
constexpr float frexp(float x, int* pw2) noexcept {
	uint32_t ex = 0x7F800000u; // exponent mask
	uint32_t ix = std::bit_cast<uint32_t>(x);
	ex &= ix;
	ix &= ~0x7F800000u;           // clear exponent
	*pw2 = (int)(ex >> 23) - 126; // compute exponent
	ix |= 0x3F000000u;            // insert exponent +1 in x
	return std::bit_cast<float>(ix);
}
/// frexp,  Get significand and exponent
constexpr double frexp(double x, int* pw2) noexcept {
	uint64_t ex = 0x7ff0000000000000; // exponent mask
	uint64_t ix = std::bit_cast<uint64_t>(x);
	ex &= ix;
	ix &= ~0x7ff0000000000000;     // clear exponent
	*pw2 = (int)(ex >> 52) - 1022; // compute exponent
	ix |= 0x3fe0000000000000;      // insert exponent +1 in x
	return std::bit_cast<double>(ix);
}
#endif
/// Rounds x toward zero, returning the nearest integral value that is not larger in magnitude than x.
__forceinline constexpr float trunc(float x) noexcept {
	float fr = x - (int32_t)x; //TODO: float int conversion !
	return (isinf(x) || fabs(x) >= (float)(INT64_C(1) << 23)) ? x : copysign(x - fr, x);
}
/// Rounds x toward zero, returning the nearest integral value that is not larger in magnitude than x.
__forceinline constexpr double trunc(double x) noexcept {
	double fr = x - (double)(INT64_C(1) << 31) * (int32_t)(x * (1.0 / (INT64_C(1) << 31)));
	fr = fr - (int32_t)fr;
	return (isinf(x) || fabs(x) >= (double)(INT64_C(1) << 52)) ? x : copysign(x - fr, x);
}

///  
__forceinline constexpr float floor(float x) noexcept {
	float fr = x - (int32_t)x;
	fr = fr < 0.0 ? fr + 1.0f : fr;
	return (isinf(x) || fabs(x) >= (float)(INT64_C(1) << 23)) ? x : copysign(x - fr, x);
}
///  
__forceinline constexpr double floor(double x) noexcept {
	double fr = x - (double)(INT64_C(1) << 31) * (int32_t)(x * (1.0 / (INT64_C(1) << 31)));
	fr = fr - (int32_t)fr;
	fr = fr < 0.0 ? fr + 1.0 : fr;
	return (isinf(x) || fabs(x) >= (double)(INT64_C(1) << 52)) ? x : copysign(x - fr, x);
}

///  
__forceinline constexpr float ceil(float x) noexcept {
	float fr = x - (int32_t)x;
	fr = fr <= 0.0 ? fr : fr - 1.0f;
	return (isinf(x) || fabs(x) >= float(INT64_C(1) << 23)) ? x : copysign(x - fr, x);
}
///  
__forceinline constexpr double ceil(double x) noexcept {
	double fr = x - double(INT64_C(1) << 31) * (int32_t)(x * (1.0 / (INT64_C(1) << 31)));
	fr = fr - (int32_t)fr;
	fr = fr <= 0.0 ? fr : fr - 1.0;
	return (isinf(x) || fabs(x) >= double(INT64_C(1) << 52)) ? x : copysign(x - fr, x);
}

/// Returns the integral value that is nearest to x, with halfway cases rounded away from zero.
__forceinline constexpr double round(double d) noexcept {
	double x = d + 0.5;
	double fr = x - double(INT64_C(1) << 31) * (int32_t)(x * (1.0 / (INT64_C(1) << 31)));
	fr = fr - (int32_t)fr;
	if (fr == 0.0 && x <= 0.0) { x -= 1.0; } // x--;
	fr = fr < 0.0 ? fr + 1.0 : fr;
	x = d == 0.49999999999999994449 ? 0.0 : x;  // nextafter(0.5, 0)
	return (isinf(d) || fabs(d) >= double(INT64_C(1) << 52)) ? d : copysign(x - fr, d);
}
/// Returns the integral value that is nearest to x, with halfway cases rounded away from zero.
__forceinline constexpr float round(float d) noexcept {
	float x = d + 0.5f;
	float fr = x - (int32_t)x;
	if (fr == 0.0f && x <= 0.0f) { x -= 1.0f; } //TODO: x--; causes compile error here !
	fr = fr < 0.0f ? fr + 1.0f : fr;
	x = (d == 0.4999999701976776123f) ? 0.0f : x;  // nextafterf(0.5, 0)
	return (isinf(d) || fabs(d) >= float(INT64_C(1) << 23)) ? d : copysign(x - fr, d);
}


/// a * b + c  in one go
__forceinline constexpr float fma(float a, float b, float c) noexcept {
    if (std::is_constant_evaluated()) {
		CX_CLANG_PRAGMA(clang fp contract(fast))
		return a * b + c;
    } else {
        return std::fma(a,b,c);
    }
}
/// a * b + c  in one go
__forceinline constexpr double fma(double a, double b, double c) noexcept {
	if (std::is_constant_evaluated()) {
		CX_CLANG_PRAGMA(clang fp contract(fast))
		return a * b + c;
	}
	else {
		return std::fma(a, b, c);
	}
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// SQRT, RSQRT
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail {

/// sqrtf(f)   do not work with subnormal numbers 
// CR:     -1 for reciprocal square root, 1 for square root
template<int CR>
__forceinline constexpr float sqrt_f_u2(float f) noexcept {
	// http://www.lomont.org/Math/Papers/2003/InvSqrt.pdf

    constexpr bool todo = true; //TODO: !!!

    float q = 0.5f;
    if constexpr (todo) { 
        if (f == 0.0f) [[unlikely]] { return f; }
		else if (f < 5.2939559203393770e-23f) [[unlikely]] {
			f *= 1.8889465931478580e+22f;
			q = 7.2759576141834260e-12f * 0.5f;
		}
		else if (f > 1.8446744073709552e+19f) [[unlikely]] {
			f *= 5.4210108624275220e-20f;
			q = 4294967296.0f * 0.5f;
		}
    }

	auto xhalf = (0.5f) * f;

    uint32_t i;
    //if constexpr (CR == 1) { // square root
	    i = std::bit_cast<uint32_t>(f + 1e-45f); // add really small number 
	    i = 0x5f375a99 - (i >> 1); // i = 0x5f375a86 - (i >> 1);
	//}
	//else if constexpr (CR == -1) {  // reciprocal square root
	//    i = std::bit_cast<uint32_t>(f); // add really small number 
	//    i = 0x5f3759df - (i >> 1);
	//}
	auto x = std::bit_cast<float>(i);

	// Newton steps, repeating this increases accuracy
	x = x * (1.5f - xhalf * x * x);
	x = x * (1.5f - xhalf * x * x);
	x = x * (1.5f - xhalf * x * x);
	// x contains the inverse sqrt
   
    if constexpr (todo) { 
        if constexpr (CR == 1) { // square root
		    x = x * f; // Multiply the inverse sqrt by the input to get the sqrt
		    const double dx = double(x);
		    const double d2 = (double(f) + dx * dx) * 1.0 / dx;
		    float r = float(d2) * q;
            // ret = (f == 0.0f) ? f : ret;

			//TODO: ulp correction, do not work for very big values !
			if (q == 0.5f) {
				auto diff = std::bit_cast<uint32_t>(f) - std::bit_cast<uint32_t>(r * r);
				r = std::bit_cast<float>(std::bit_cast<uint32_t>(r) + (diff / 2));
			}

            return r;
		}
		else if constexpr (CR == -1) {  // reciprocal square root
			return x; //TODO: do not work 
		}
    } else {
		if constexpr (CR == 1) { // square root
			return f * x; // Multiply the inverse sqrt by the input to get the sqrt
        }
		else if constexpr (CR == -1) {  // reciprocal square root
			return x; //TODO: do not work 
		}
    }
}

/// 1.0f / sqrtf(f)   do not work with subnormal numbers 
__forceinline constexpr float rsqrt_f_u1(float x) //TODO: ulp error?
{
	if (std::is_constant_evaluated()) {
		if (x <= 0.0f) [[unlikely]] return NAN; // handle x < 0
	}
	else {
		x = (x > 0.0f) ? x : NAN; // handle x < 0
	}

	constexpr auto SQRT_MAGIC_F = 0x5f3759df;
	const float xhalf = 0.5f * x;

	int32_t i = std::bit_cast<int32_t>(x);
	i = SQRT_MAGIC_F - (i >> 1);  // gives initial guess
	float nx = std::bit_cast<float>(i);

	// Newton step, repeating increases accuracy 
#if 1
	nx = nx * (1.5f - xhalf * nx * nx); // 16024 ulp  
	nx = nx * (1.5f - xhalf * nx * nx); // 38 ulp
	nx = nx * (1.5f - xhalf * nx * nx); //? ulp
	
	double dnx = double(nx);
	nx = float(dnx * (1.5 - 0.5 * double(x) * dnx * dnx)); //? ulp
#else
	double dnx = double(nx);
	double dx = double(x);
	dnx = dnx * (1.5 - 0.5 * dx * dnx * dnx); 
	dnx = dnx * (1.5 - 0.5 * dx * dnx * dnx);
	dnx = dnx * (1.5 - 0.5 * dx * dnx * dnx);
	dnx = dnx * (1.5 - 0.5 * dx * dnx * dnx);
	nx = float(dnx);
#endif
	return nx;
}

/// sqrt(d)   do not work with subnormal numbers 
// CR:     -1 for reciprocal square root, 1 for square root
template<int CR>
__forceinline constexpr double sqrt_u2(double f) noexcept {
	// http://www.lomont.org/Math/Papers/2003/InvSqrt.pdf
	auto xhalf = (0.5) * f;

	auto i = std::bit_cast<uint64_t>(f + 1e-320); // add really small number 
	i = 0x5fe6eb50c7b537a9ULL - (i >> 1);
	auto x = std::bit_cast<double>(i);

	// Newton steps, repeating this increases accuracy
	x = x * ((1.5) - xhalf * x * x);
	x = x * ((1.5) - xhalf * x * x);
	x = x * ((1.5) - xhalf * x * x);
	x = x * ((1.5) - xhalf * x * x); // 3 ulp
	// x = x * ((1.5) - xhalf * x * x); // 2 ulp
	// x contains the inverse sqrt

    if constexpr (CR == 1) { // square root
	    return f * x; // Multiply the inverse sqrt by the input to get the sqrt
    }
	else if constexpr (CR == -1) {  // reciprocal square root
		return x;
	}
}

//=--------------------------------------------------------------------------------------------------------------------
/// 1.0 / sqrt(d)   do not work with subnormal numbers 
__forceinline constexpr double rsqrt_u2(double f)
{
	// http://www.lomont.org/Math/Papers/2003/InvSqrt.pdf
	auto xhalf = (0.5) * f;

	auto i = std::bit_cast<uint64_t>(f + 1e-320); // add really small number 
	i = 0x5fe6eb50c7b537a9ULL - (i >> 1);
	auto x = std::bit_cast<double>(i);

	// Newton steps, repeating this increases accuracy
	x = x * ((1.5) - xhalf * x * x);
	x = x * ((1.5) - xhalf * x * x);
	x = x * ((1.5) - xhalf * x * x);

	x = x * ((1.5) - xhalf * x * x); // 3 ulp
	
	//? x = x * ((1.5) - xhalf * x * x); // 2 ulp
	// x contains the inverse sqrt
	return x;
}

} // namespace detail


/// √x, sqrtf(x) 
__forceinline constexpr float sqrtf(float x) noexcept {
	if (std::is_constant_evaluated()) { 
		return cx::detail::sqrt_f_u2<1>(x); }
    else {
        return std::sqrt(x);
	}
}
/// √x, sqrt(x) 
__forceinline constexpr float sqrt(float x) noexcept {
	if (std::is_constant_evaluated()) { 
		return cx::detail::sqrt_f_u2<1>(x); }
	else {
        return std::sqrt(x);
	}
}
/// √x, sqrt(x)
__forceinline constexpr double sqrt(double x) noexcept {
	if (std::is_constant_evaluated()) { return cx::detail::sqrt_u2<1>(x); }
	else {
        return std::sqrt(x);
	}
}

/// 1/√x, 1/sqrtf(x) 
__forceinline constexpr float rsqrtf(float x) noexcept {
	if (std::is_constant_evaluated()) { 
		return detail::rsqrt_f_u1(x);
		// return cx::detail::sqrt_f_u2<-1>(x);
	}
	else {
		return 1.0f/std::sqrt(x);
	}
}
/// 1/√x, 1/sqrt(x) 
__forceinline constexpr float rsqrt(float x) noexcept {
	if (std::is_constant_evaluated()) { 
		return detail::rsqrt_f_u1(x);
		// return cx::detail::sqrt_f_u2<-1>(x);
	}
	else {
		return 1.0f/std::sqrt(x);
	}
}
/// 1/√x, 1/sqrt(x)
__forceinline constexpr double rsqrt(double x) noexcept {
	if (std::is_constant_evaluated()) { 
		//return cx::detail::sqrt_u2<-1>(x);
		return detail::rsqrt_u2(x);
	}
	else {
		return 1.0/std::sqrt(x);
	}
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// CBRT, RCBRT
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail {
/// Calculate the cube root of the input argument. 
//  ulp:  5 | ave_ulp: 0.9  //  5.7 ns/op  Fast !!!
// CR:     -1 for reciprocal cube root, 1 for cube root
template<int CR>
constexpr float cbrt_f_u5(float x) noexcept {
	constexpr float k1 = 1.752319676f;
	constexpr float k2 = 1.2509524245f;
	constexpr float k3 = 0.5093818292f;

	uint32_t i = std::bit_cast<uint32_t>(x);
	uint32_t sign = i & 0x80000000u; // signbit
	i &= 0x7FFFFFFFu; // abs

	i = 0x548c2b4bu - i / 3;
	float y = std::bit_cast<float>(i ^ sign);

	float c = x * y * y * y;
	y = y * (k1 - c * (k2 - k3 * c));

	c = 1.0f - x * y * y * y; // fmaf
	y = y * (1.0f + 0.333333333333f * c); // fmaf

    if constexpr (CR == 1) {        // cube root
	    return x * y * y;           // convert 1/cbrt(x) to cbrt(x) !
    }
	else if constexpr (CR == -1) {  // reciprocal cube root
		return y;
	}
    else if constexpr (CR == 2) {   // cube root squared
        return x * y;
    }
}


/// Calculate the cube root of the input argument. 
//  round-trip: max ulp 14  ulp_histo: | 0: 19.2% | 1: 22.7% | 2: 22.8% | 3: 17.2% | 4: 9.13% | 5: 4.32% | 6: 2.98% | 7: 1.14% | 8: 0.284% | 9: 0.196% | 10: 0.0535% | 11: 0.0021% | 12: 0.00164% | 13: 2.9e-06% | 14: 1.3e-06% |
//  ulp: 5 | ave_ulp: 0.81  //  8.19  ns/op
// CR:     -1 for reciprocal cube root, 1 for cube root
template<int CR>
constexpr double cbrt_u5(double x) noexcept {
	constexpr double k1 = 1.7523196763699390234751750023038468;
	constexpr double k2 = 1.2509524245066599988510127816507199;
	constexpr double k3 = 0.50938182920440939104272244099570921;

	constexpr double one_third = (1.0 / 3.0);  // = 0.3333333333333333;
	// constexpr double four_third = (4.0 / 3.0); // = 1.3333333333333333; 

	uint64_t i = std::bit_cast<uint64_t>(x);
	uint64_t sign = i & 0x8000000000000000ull; // signbit   sign = i & (1<<63);
	i &= 0x7FFFFFFFFFFFFFFFull; // abs  // i &= (~(1<<63));

	/// magic = approx + in/3
	/// Julia: magic = reinterpret(UInt64,1.0/cbrt(1000.0)) + div(reinterpret(UInt64,1000.0),3)
	/// Julia: 0x553c611bfd44f307 = reinterpret(UInt64,0.091) + div(reinterpret(UInt64,1000.0),3)
	i = 0x553C3000000584f0u - i / 3; // aprox_div3(i);

	double y = std::bit_cast<double>(i ^ sign);

	double c = (x * y) * (y * y);
	y = y * (k1 - c * (k2 - k3 * c));

	c = 1.0 - (x * y) * (y * y); // fmaf
	y = y * (1.0 + one_third * c); // fmaf

	y += one_third * (y - (x * y) * (y * y) * y); // round: 12 ulp ?

    if constexpr (CR == 1) {        // cube root
	    return x * y * y;           // convert 1/cbrt(x) to cbrt(x) !
    }
    else if constexpr (CR == -1) {  // reciprocal cube root
        return y;
    }
    else if constexpr (CR == 2) {   // cube root squared
        return x * y;
    }
}

} // namespace detail

namespace detail {
/*
* Based on      https://github.com/vectorclass/version2
* Author:       Agner Fog
*/

/******************************************************************************
Define NAN payload values
******************************************************************************/
constexpr auto NAN_LOG = 0x101;  // logarithm for x<0
constexpr auto NAN_POW = 0x102;  // negative number raised to non-integer power
constexpr auto NAN_HYP = 0x104;  // acosh for x<1 and atanh for abs(x)>1


/******************************************************************************
Define mathematical constants
******************************************************************************/
constexpr auto VM_PI       = 3.14159265358979323846  ;         // pi
constexpr auto VM_PI_2     = 1.57079632679489661923  ;         // pi / 2
constexpr auto VM_PI_4     = 0.785398163397448309616 ;         // pi / 4
constexpr auto VM_SQRT2    = 1.41421356237309504880  ;         // sqrt(2)
constexpr auto VM_LOG2E    = 1.44269504088896340736  ;         // 1/log(2)

constexpr auto VM_LOG10E   = 0.434294481903251827651 ;         // 1/log(10)
constexpr auto VM_LOG210   = 3.321928094887362347808 ;         // log2(10)
constexpr auto VM_LN2      = 0.693147180559945309417 ;         // log(2)
constexpr auto VM_LN10     = 2.30258509299404568402  ;         // log(10)
constexpr auto VM_SMALLEST_NORMAL  = 2.2250738585072014E-308;  // smallest normal number, double
constexpr auto VM_SMALLEST_NORMALF = 1.17549435E-38f;          // smallest normal number, float


// Multiply and add
template<class VTYPE>
static constexpr VTYPE mul_add(VTYPE const a, VTYPE const b, VTYPE const c) {
#if 0 // def __FMA__
    return _mm_fmadd_pd(a, b, c);
#else
    CX_CLANG_PRAGMA(clang fp contract(fast))
    return a * b + c;
#endif
}

// Multiply and inverse subtract
template<class VTYPE>
static constexpr VTYPE nmul_add(VTYPE const a, VTYPE const b, VTYPE const c) {
#if 0 // def __FMA__
    return _mm_fnmadd_pd(a, b, c);
#else
    CX_CLANG_PRAGMA(clang fp contract(fast))
    return c - a * b;
#endif
}

// Multiply and subtract
template<class VTYPE>
static constexpr VTYPE mul_sub(VTYPE const a, VTYPE const b, VTYPE const c) {
#if 0 // def __FMA__
	return _mm_fmsub_ps(a, b, c);
#else
    CX_CLANG_PRAGMA(clang fp contract(fast))
	return a * b - c;
#endif
}

// template <typedef VECTYPE, typedef CTYPE>
template <class VTYPE, class CTYPE>
static constexpr VTYPE polynomial_2(VTYPE const x, CTYPE c0, CTYPE c1, CTYPE c2) {
	// calculates polynomial c2*x^2 + c1*x + c0
	// VTYPE may be a vector type, CTYPE is a scalar type
	VTYPE x2 = x * x;
	return mul_add(x2, c2, mul_add(x, c1, c0)); // return = x2 * c2 + (x * c1 + c0);
}

template<class VTYPE, class CTYPE>
static constexpr VTYPE polynomial_3(VTYPE const x, CTYPE c0, CTYPE c1, CTYPE c2, CTYPE c3) {
	// calculates polynomial c3*x^3 + c2*x^2 + c1*x + c0
	// VTYPE may be a vector type, CTYPE is a scalar type
	VTYPE x2 = x * x;
	return mul_add(mul_add(c3, x, c2), x2, mul_add(c1, x, c0)); // return (c2 + c3*x)*x2 + (c1*x + c0);
}

template<class VTYPE, class CTYPE>
static constexpr VTYPE polynomial_4(VTYPE const x, CTYPE c0, CTYPE c1, CTYPE c2, CTYPE c3, CTYPE c4) {
	// calculates polynomial c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
	// VTYPE may be a vector type, CTYPE is a scalar type
	VTYPE x2 = x * x;
	VTYPE x4 = x2 * x2;
	return mul_add(mul_add(c3, x, c2), x2, mul_add(c1, x, c0) + c4 * x4); // return (c2+c3*x)*x2 + ((c0+c1*x) + c4*x4);
}

template<class VTYPE, class CTYPE>
static constexpr VTYPE polynomial_4n(VTYPE const x, CTYPE c0, CTYPE c1, CTYPE c2, CTYPE c3) {
	// calculates polynomial 1*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
	// VTYPE may be a vector type, CTYPE is a scalar type
	VTYPE x2 = x * x;
	VTYPE x4 = x2 * x2;
	return mul_add(mul_add(c3, x, c2), x2, mul_add(c1, x, c0) + x4); // return (c2+c3*x)*x2 + ((c0+c1*x) + x4);
}

template<class VTYPE, class CTYPE>
static constexpr VTYPE polynomial_5(VTYPE const x, CTYPE c0, CTYPE c1, CTYPE c2, CTYPE c3, CTYPE c4, CTYPE c5) {
	// calculates polynomial c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
	// VTYPE may be a vector type, CTYPE is a scalar type
	VTYPE x2 = x * x;
	VTYPE x4 = x2 * x2;
	return mul_add(mul_add(c3, x, c2), x2, mul_add(mul_add(c5, x, c4), x4, mul_add(c1, x, c0))); // return (c2+c3*x)*x2 + ((c4+c5*x)*x4 + (c0+c1*x));
}

template<class VTYPE, class CTYPE>
static constexpr VTYPE polynomial_5n(VTYPE const x, CTYPE c0, CTYPE c1, CTYPE c2, CTYPE c3, CTYPE c4) {
	// calculates polynomial 1*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
	// VTYPE may be a vector type, CTYPE is a scalar type
	VTYPE x2 = x * x;
	VTYPE x4 = x2 * x2;
	return mul_add(mul_add(c3, x, c2), x2, mul_add(c4 + x, x4, mul_add(c1, x, c0))); // return (c2+c3*x)*x2 + ((c4+x)*x4 + (c0+c1*x));
}

template<class VTYPE, class CTYPE>
static constexpr VTYPE polynomial_6(VTYPE const x, CTYPE c0, CTYPE c1, CTYPE c2, CTYPE c3, CTYPE c4, CTYPE c5, CTYPE c6) {
	// calculates polynomial c6*x^6 + c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
	// VTYPE may be a vector type, CTYPE is a scalar type
	VTYPE x2 = x * x;
	VTYPE x4 = x2 * x2;
	return mul_add(mul_add(c6, x2, mul_add(c5, x, c4)), x4, mul_add(mul_add(c3, x, c2), x2, mul_add(c1, x, c0))); // return  (c4+c5*x+c6*x2)*x4 + ((c2+c3*x)*x2 + (c0+c1*x));
}

template<class VTYPE, class CTYPE>
static constexpr VTYPE polynomial_6n(VTYPE const x, CTYPE c0, CTYPE c1, CTYPE c2, CTYPE c3, CTYPE c4, CTYPE c5) {
	// calculates polynomial 1*x^6 + c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
	// VTYPE may be a vector type, CTYPE is a scalar type
	VTYPE x2 = x * x;
	VTYPE x4 = x2 * x2;
	return mul_add(mul_add(c5, x, c4 + x2), x4, mul_add(mul_add(c3, x, c2), x2, mul_add(c1, x, c0))); // return  (c4+c5*x+x2)*x4 + ((c2+c3*x)*x2 + (c0+c1*x));
}

template<class VTYPE, class CTYPE>
static constexpr VTYPE polynomial_7(VTYPE const x, CTYPE c0, CTYPE c1, CTYPE c2, CTYPE c3, CTYPE c4, CTYPE c5, CTYPE c6, CTYPE c7) {
	// calculates polynomial c7*x^7 + c6*x^6 + c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
	// VTYPE may be a vector type, CTYPE is a scalar type
	VTYPE x2 = x * x;
	VTYPE x4 = x2 * x2;
	//return  ((c6+c7*x)*x2 + (c4+c5*x))*x4 + ((c2+c3*x)*x2 + (c0+c1*x));
	return mul_add(mul_add(mul_add(c7, x, c6), x2, mul_add(c5, x, c4)), x4, mul_add(mul_add(c3, x, c2), x2, mul_add(c1, x, c0)));
}

template<class VTYPE, class CTYPE>
static constexpr VTYPE polynomial_8(VTYPE const x, CTYPE c0, CTYPE c1, CTYPE c2, CTYPE c3, CTYPE c4, CTYPE c5, CTYPE c6, CTYPE c7, CTYPE c8) {
	// calculates polynomial c8*x^8 + c7*x^7 + c6*x^6 + c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
	// VTYPE may be a vector type, CTYPE is a scalar type
	VTYPE x2 = x * x;
	VTYPE x4 = x2 * x2;
	VTYPE x8 = x4 * x4;
	//return  ((c6+c7*x)*x2 + (c4+c5*x))*x4 + (c8*x8 + (c2+c3*x)*x2 + (c0+c1*x));
	return mul_add(mul_add(mul_add(c7, x, c6), x2, mul_add(c5, x, c4)), x4,
		mul_add(mul_add(c3, x, c2), x2, mul_add(c1, x, c0) + c8 * x8));
}

template<class VTYPE, class CTYPE>
static constexpr VTYPE polynomial_9(VTYPE const x, CTYPE c0, CTYPE c1, CTYPE c2, CTYPE c3, CTYPE c4, CTYPE c5, CTYPE c6, CTYPE c7, CTYPE c8, CTYPE c9) {
	// calculates polynomial c9*x^9 + c8*x^8 + c7*x^7 + c6*x^6 + c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
	// VTYPE may be a vector type, CTYPE is a scalar type
	VTYPE x2 = x * x;
	VTYPE x4 = x2 * x2;
	VTYPE x8 = x4 * x4;
	//return  (((c6+c7*x)*x2 + (c4+c5*x))*x4 + (c8+c9*x)*x8) + ((c2+c3*x)*x2 + (c0+c1*x));
	return mul_add(mul_add(c9, x, c8), x8, mul_add(
		mul_add(mul_add(c7, x, c6), x2, mul_add(c5, x, c4)), x4,
		mul_add(mul_add(c3, x, c2), x2, mul_add(c1, x, c0))));
}

template<class VTYPE, class CTYPE>
static constexpr VTYPE polynomial_10(VTYPE const x, CTYPE c0, CTYPE c1, CTYPE c2, CTYPE c3, CTYPE c4, CTYPE c5, CTYPE c6, CTYPE c7, CTYPE c8, CTYPE c9, CTYPE c10) {
	// calculates polynomial c10*x^10 + c9*x^9 + c8*x^8 + c7*x^7 + c6*x^6 + c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
	// VTYPE may be a vector type, CTYPE is a scalar type
	VTYPE x2 = x * x;
	VTYPE x4 = x2 * x2;
	VTYPE x8 = x4 * x4;
	//return  (((c6+c7*x)*x2 + (c4+c5*x))*x4 + (c8+c9*x+c10*x2)*x8) + ((c2+c3*x)*x2 + (c0+c1*x));
	return mul_add(mul_add(x2, c10, mul_add(c9, x, c8)), x8,
		mul_add(mul_add(mul_add(c7, x, c6), x2, mul_add(c5, x, c4)), x4,
			mul_add(mul_add(c3, x, c2), x2, mul_add(c1, x, c0))));
}

template<class VTYPE, class CTYPE>
static constexpr VTYPE polynomial_13m(VTYPE const x, CTYPE c2, CTYPE c3, CTYPE c4, CTYPE c5, CTYPE c6, CTYPE c7, CTYPE c8, CTYPE c9, CTYPE c10, CTYPE c11, CTYPE c12, CTYPE c13) {
	// calculates polynomial c13*x^13 + c12*x^12 + ... + x + 0
	// VTYPE may be a vector type, CTYPE is a scalar type
	VTYPE x2 = x * x;
	VTYPE x4 = x2 * x2;
	VTYPE x8 = x4 * x4;
	// return  ((c8+c9*x) + (c10+c11*x)*x2 + (c12+c13*x)*x4)*x8 + (((c6+c7*x)*x2 + (c4+c5*x))*x4 + ((c2+c3*x)*x2 + x));
	return mul_add(
		mul_add(mul_add(c13, x, c12), x4, mul_add(mul_add(c11, x, c10), x2, mul_add(c9, x, c8))), x8,
		mul_add(mul_add(mul_add(c7, x, c6), x2, mul_add(c5, x, c4)), x4, mul_add(mul_add(c3, x, c2), x2, x)));
}


static constexpr int32_t reinterpret_i(float x) noexcept { return std::bit_cast<int32_t>(x); }

static constexpr uint32_t reinterpret_u(int32_t x) noexcept { return std::bit_cast<uint32_t>(x); } //TODO: unsigned
static constexpr uint32_t reinterpret_u(float x) noexcept { return std::bit_cast<uint32_t>(x); } //TODO: unsigned

static constexpr float reinterpret_f(int32_t x) noexcept { return std::bit_cast<float>(x); }
static constexpr float reinterpret_f(uint32_t x) noexcept { return std::bit_cast<float>(x); }


static constexpr int64_t reinterpret_i(double x) noexcept { return std::bit_cast<int64_t>(x); }

static constexpr uint64_t reinterpret_u(int64_t x) noexcept { return std::bit_cast<uint64_t>(x); } //TODO: unsigned
static constexpr uint64_t reinterpret_u(double x) noexcept { return std::bit_cast<uint64_t>(x); } //TODO: unsigned

// static constexpr double reinterpret_d(int32_t x) noexcept { return std::bit_cast<double>(int64_t(x)); } //? Is this correct ?

static constexpr double reinterpret_d(int64_t x) noexcept { return std::bit_cast<double>(x); }
static constexpr double reinterpret_d(uint64_t x) noexcept { return std::bit_cast<double>(x); }

// Function is_finite: gives true for elements that are normal, denormal or zero, false for INF and NAN
static constexpr bool is_finite(float a) noexcept { 
    auto t1 = std::bit_cast<uint32_t>(a);    // reinterpret as 32-bit integer
    auto t2 = t1 << 1;                // shift out sign bit
    bool t3 = uint32_t(t2 & 0xFF000000) != 0xFF000000; // exponent field is not all 1s
	return t3;
}

// Function is_finite: gives true for elements that are normal, denormal or zero, false for INF and NAN
static constexpr bool is_finite(double a) noexcept { 
    auto t1 = std::bit_cast<uint64_t>(a);    // reinterpret as integer
    auto t2 = t1 << 1;                // shift out sign bit
    uint64_t t3 = 0xFFE0000000000000ll;   // exponent mask
    bool t4 = uint64_t(t2 & t3) != t3;  // exponent field is not all 1s
	return t4;
}

static constexpr float sign_combine(float a, float b) noexcept {
	return std::bit_cast<float>(
		std::bit_cast<uint32_t>(a) ^ (std::bit_cast<uint32_t>(b) & std::bit_cast<uint32_t>(-0.0f))
		);
}
static constexpr double sign_combine(double a, double b) noexcept {
    return std::bit_cast<double>(
        std::bit_cast<uint64_t>(a) ^ (std::bit_cast<uint64_t>(b) & std::bit_cast<uint64_t>(-0.0))
        );
}

// horizontal_and. Returns true if all elements are true
static constexpr bool horizontal_and(bool a) noexcept { return a; }

// horizontal_or. Returns true if at least one element is true
static constexpr bool horizontal_or(bool a) noexcept { return a; }

// function andnot: a & ~ b
static constexpr bool andnot(bool a, bool b) noexcept {  return a && (!b); }

static constexpr __forceinline double bit_or(double a, double b) noexcept { return std::bit_cast<double>( std::bit_cast<uint64_t>(a) | std::bit_cast<uint64_t>(b) ); }
static constexpr __forceinline float  bit_or(float a, float b) noexcept   { return std::bit_cast<float>( std::bit_cast<uint32_t>(a) | std::bit_cast<uint32_t>(b) ); }

static constexpr double bit_or(bool m, double b) noexcept = delete;
static constexpr float  bit_or(bool m, float b) noexcept = delete;


static constexpr __forceinline double bit_and(double a, double b) noexcept { return std::bit_cast<double>(std::bit_cast<uint64_t>(a) & std::bit_cast<uint64_t>(b)); }
static constexpr __forceinline float  bit_and(float a, float b) noexcept   { return std::bit_cast<float>(std::bit_cast<uint32_t>(a) & std::bit_cast<uint32_t>(b)); }

static constexpr __forceinline double bit_and(bool m, double b) noexcept { return select(m,b, 0.0); }
static constexpr __forceinline float  bit_and(bool m, float b) noexcept { return select(m,b, 0.0f); }

// template for producing quiet NAN
template <class VTYPE>
static constexpr VTYPE nan_vec(uint32_t payload = 0x100) noexcept {
    if constexpr (std::is_same_v<VTYPE, double>) {
		// n is left justified to avoid loss of NAN payload when converting to float
		return std::bit_cast<double>(0x7FF8000000000000 | uint64_t(payload) << 29);
    } 
	return std::bit_cast<float>(0x7FC00000 | (payload & 0x003FFFFF)); 
}


static constexpr bool is_nan(float a) noexcept {
    return cx::isnan(a); // a != a; is not safe with -ffinite-math-only, -ffast-math, or /fp:fast compiler option
}

static constexpr bool is_nan(double a) noexcept {
    return cx::isnan(a); // a != a; is not safe with -ffinite-math-only, -ffast-math, or /fp:fast compiler option
}

static constexpr uint32_t nan_code(float x) noexcept {
    uint32_t a = reinterpret_i(x);
    uint32_t const n = 0x007FFFFF;
    return select(is_nan(x), a & n, uint32_t(0));
}

// This function returns the code hidden in a NAN. The sign bit is ignored
static constexpr uint64_t nan_code(double x) noexcept {
    uint64_t a = reinterpret_i(x);
    return select(is_nan(x), a << 12 >> (12+29), uint64_t(0));
}


template <class VTYPE>
static constexpr VTYPE infinite_vec() noexcept;

// returns +INF
template <>
constexpr double infinite_vec<double>() noexcept { return reinterpret_d(0x7FF0000000000000); }

// returns +INF
template <>
constexpr float infinite_vec<float>() noexcept { return reinterpret_f(0x7F800000); }

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
static constexpr bool is_inf(double a) noexcept {
    uint64_t t1 = std::bit_cast<uint64_t>(a);    // reinterpret as integer
    uint64_t t2 = t1 << 1;                // shift out sign bit
	return t2 == 0xFFE0000000000000ll; // exponent is all 1s, fraction is 0
}


// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static constexpr bool is_inf(float a) noexcept {
    uint32_t t1 = std::bit_cast<uint32_t>(a);    // reinterpret as 32-bit integer
    uint32_t t2 = t1 << 1;                // shift out sign bit
	return t2 == uint32_t(0xFF000000);    // exponent is all 1s, fraction is 0
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal (denormal)
// false for finite numbers, NAN and INF
static constexpr float is_zero_or_subnormal(float a) noexcept {
    uint32_t t = std::bit_cast<uint32_t>(a);     // reinterpret as 32-bit integer
	t &= 0x7F800000;                   // isolate exponent
	return t == 0;                     // exponent = 0
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal (denormal)
// false for finite numbers, NAN and INF
static constexpr double is_zero_or_subnormal(double a) noexcept {
    uint64_t t = std::bit_cast<uint64_t>(a);     // reinterpret as 32-bit integer
	t &= 0x7FF0000000000000ll;   // isolate exponent
	return t == 0;                     // exponent = 0
}

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0f, -INF and -NAN
// Note that sign_bit(Vec4f(-0.0f)) gives true, while Vec4f(-0.0f) < Vec4f(0.0f) gives false
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static constexpr bool sign_bit(float a) noexcept {
    // uint32_t t1 = std::bit_cast<uint32_t>(a);    // reinterpret as 32-bit integer
	// uint32_t t2 = t1 >> 31;               // extend sign bit
	// return t2 != 0;
	uint32_t i = std::bit_cast<uint32_t>(a);
	uint32_t sign = i & 0x80000000u; // signbit
    return sign;
}

static_assert(sign_bit(0.0f) == false);
static_assert(sign_bit(-0.0f) == true);

static_assert(sign_bit(1.0f) == false);
static_assert(sign_bit(-1.0f) == true);

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0f, -INF and -NAN
// Note that sign_bit(Vec4f(-0.0f)) gives true, while Vec4f(-0.0f) < Vec4f(0.0f) gives false
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static constexpr bool sign_bit(double a) noexcept {
	// uint64_t t1 = std::bit_cast<uint64_t>(a);    // reinterpret as 32-bit integer
	// uint64_t t2 = t1 >> 31;               // extend sign bit
	// return t2 != 0;
	uint64_t i = std::bit_cast<uint64_t>(a);
	uint64_t sign = i & 0x8000000000000000ull; // signbit   sign = i & (1<<63);
    return sign;
}

static_assert(sign_bit(0.0) == false);
static_assert(sign_bit(-0.0) == true);

static_assert(sign_bit(1.0) == false);
static_assert(sign_bit(-1.0) == true);

static constexpr double round(double a) noexcept {
	constexpr auto _TWO_TO_52 = 4503599627370496.0;
	double fa = fabs(a);
	double u = _TWO_TO_52 + fa;
	if (fa >= _TWO_TO_52) { u = a; }
	else {
		u = u - _TWO_TO_52;
		u = copysign(u, a);
	}
	return u;
} //TODO: !!!

static constexpr float round(float x) noexcept { 
    return float(x < 0.0f ? int32_t(x - 0.5f) : int32_t(x + 0.5f));  //TODO: integer conversion !
} //TODO: !!!


static constexpr int64_t roundi(double const a) noexcept { return int64_t(round(a)); } //TODO: Optimize !!!

static constexpr int32_t roundi(float a) noexcept { return int32_t(round(a)); } //TODO: Optimize !!!

} // namespace detail 

namespace detail {
#if 0
// cube root template, double precision
// template parameters:
// VTYPE:  f.p. vector type
// CR:     -1 for reciprocal cube root, 1 for cube root, 2 for cube root squared
template<typename VTYPE, int CR>
static constexpr VTYPE cbrt_d(VTYPE const x) {
	const int iter = 7;     // iteration count of x^(-1/3) loop
	int i;
	typedef decltype(x < x) BVTYPE;              // boolean vector type
	typedef decltype(roundi(x)) ITYPE64;         // 64 bit integer vector type

	BVTYPE underflow;
	ITYPE64 q1(0x5540000000000000ULL);           // exponent bias
	ITYPE64 q2(0x0005555500000000ULL);           // exponent multiplier for 1/3
	ITYPE64 q3(0x0010000000000000ULL);           // denormal limit

	VTYPE  xa, xa3, a, a2;
	const double one_third = 1.0 / 3.0;
	const double four_third = 4.0 / 3.0;

	xa = abs(x);
	xa3 = one_third * xa;


    if constexpr (std::is_same_v<VTYPE, double>) { //TODO: ??? 
        //using ITYPE32 = int32_t;

        // multiply exponent by -1/3
		ITYPE64 m1 = reinterpret_i(xa);
		ITYPE64 m2 = 0x553ebeeeeeefeeeeull - (m1 / 3); //! imprecise !
       // ITYPE64 m2 = q1 - (m1 >> 20)*ITYPE32(q2);
		a = reinterpret_d(m2);
		underflow = BVTYPE(m1 <= q3);       // true if denormal or zero
        
        //printf("   xa %f \n",xa);
        //printf("   %llx %llx \n",(q1),(q2));
        //printf("   %x %x \n",int32_t(q1),int32_t(q2));
        //printf("   m1 %llx %llx \n",m1, q1 - m1 / 3);
        //printf("   m2 %llx \n",m2);
        //printf("   a  %f \n",a);
    }
    else {
        typedef decltype(roundi(compress(x, x))) ITYPE32; // 32 bit integer vector type
        // multiply exponent by -1/3
		ITYPE32 m1 = reinterpret_i(xa);
		ITYPE32 m2 = ITYPE32(q1) - (m1 >> 20) * ITYPE32(q2);
		a = reinterpret_d(m2);
		underflow = BVTYPE(ITYPE64(m1) <= q3);       // true if denormal or zero
	}

	// Newton Raphson iteration. Warning: may overflow!
	for (i = 0; i < iter - 1; i++) {
		a2 = a * a;
		a = nmul_add(xa3, a2 * a2, four_third * a);  // a = four_third*a - xa3*a2*a2;
	}
	// last iteration with better precision
	a2 = a * a;
	a = mul_add(one_third, nmul_add(xa, a2 * a2, a), a); // a = a + one_third*(a - xa*a2*a2);

	if constexpr (CR == -1) {                    // reciprocal cube root
		a = select(underflow, infinite_vec<VTYPE>(), a); // generate INF if underflow
		a = select(is_inf(x), VTYPE(0), a);      // special case for INF                                                 // get sign
		a = sign_combine(a, x);                  // get sign
	}
	else if constexpr (CR == 1) {                // cube root
		a = a * a * x;
		a = select(underflow, 0., a);            // generate 0 if underflow
		a = select(is_inf(x), x, a);             // special case for INF
#ifdef SIGNED_ZERO
		a = a | (x & VTYPE(-0.0));                      // get sign of x
#endif
	}
	else if constexpr (CR == 2) {                // cube root squared
		a = a * xa;
		a = select(underflow, 0., a);            // generate 0 if underflow
		a = select(is_inf(x), xa, a);            // special case for INF
	}
	return a;
}

// cube root template, single precision
// template parameters:
// VTYPE:  f.p. vector type
// CR:     -1 for reciprocal cube root, 1 for cube root, 2 for cube root squared
template<typename VTYPE, int CR>
static constexpr VTYPE cbrt_f(VTYPE const x) {

	const int iter = 4;                          // iteration count of x^(-1/3) loop
	int i;

	typedef decltype(roundi(x)) ITYPE;           // integer vector type
	typedef decltype(x < x) BVTYPE;              // boolean vector type

	VTYPE  xa, xa3, a, a2;
	ITYPE  m1, m2;
	BVTYPE underflow;
	ITYPE  q1(0x54800000U);                      // exponent bias
	ITYPE  q2(0x002AAAAAU);                      // exponent multiplier for 1/3
	ITYPE  q3(0x00800000U);                      // denormal limit
	const  float one_third = float(1. / 3.);
	const  float four_third = float(4. / 3.);

	xa = abs(x);
	xa3 = one_third * xa;

	// multiply exponent by -1/3
	m1 = reinterpret_i(xa);
	m2 = q1 - (m1 >> 23) * q2;
	a = reinterpret_f(m2);

	underflow = BVTYPE(m1 <= q3);                // true if denormal or zero

	// Newton Raphson iteration
	for (i = 0; i < iter - 1; i++) {
		a2 = a * a;
		a = nmul_add(xa3, a2 * a2, four_third * a);  // a = four_third*a - xa3*a2*a2;
	}
	// last iteration with better precision
	a2 = a * a;
	a = mul_add(one_third, nmul_add(xa, a2 * a2, a), a); //a = a + one_third*(a - xa*a2*a2);

	if constexpr (CR == -1) {                    // reciprocal cube root
		// generate INF if underflow
		a = select(underflow, infinite_vec<VTYPE>(), a);
		a = select(is_inf(x), VTYPE(0), a);      // special case for INF                                                 // get sign
		a = sign_combine(a, x);
	}
	else if constexpr (CR == 1) {                // cube root
		a = a * a * x;
		a = select(underflow, 0.0f, a);           // generate 0 if underflow
		a = select(is_inf(x), x, a);             // special case for INF
#ifdef SIGNED_ZERO
		a = a | (x & VTYPE(-0.0f));                     // get sign of x
#endif
	}
	else if constexpr (CR == 2) {                // cube root squared
		a = a * xa;                              // abs only to fix -INF
		a = select(underflow, 0.0f, a);            // generate 0 if underflow
		a = select(is_inf(x), xa, a);            // special case for INF
	}
	return a;
}
#endif
} // namespace detail 

/// ∛x, cbrt(x) 
constexpr float cbrtf(float x) noexcept { return detail::cbrt_f_u5<1>(x); }
/// ∛x, cbrt(x) 
constexpr float cbrt(float x) noexcept { return detail::cbrt_f_u5<1>(x); }
/// ∛x, cbrt(x) 
constexpr double cbrt(double x) noexcept {	return detail::cbrt_u5<1>(x); }

/// 1/∛x, 1/cbrt(x) 
constexpr float rcbrtf(float x) noexcept { return detail::cbrt_f_u5<-1>(x); }
/// 1/∛x, 1/cbrt(x) 
constexpr float rcbrt(float x) noexcept { return detail::cbrt_f_u5<-1>(x); }
/// 1/∛x, 1/cbrt(x) 
constexpr double rcbrt(double x) noexcept { return detail::cbrt_u5<-1>(x); }

/// (∛x)²,  cbrt(x)²
constexpr float square_cbrtf(float x) noexcept { return detail::cbrt_f_u5<2>(x); }
/// (∛x)²,  cbrt(x)²
constexpr float square_cbrt(float x) noexcept { return detail::cbrt_f_u5<2>(x); }
/// (∛x)²,  cbrt(x)²
constexpr double square_cbrt(double x) noexcept { return detail::cbrt_u5<2>(x); }

namespace detail {

/// "An Improved Algorithm for hypot(a,b)"  https://arxiv.org/abs/1904.09481
template<typename T>
constexpr T hypot_fma(T x, T y) noexcept {
	// Return Inf if either or both inputs is Inf (Compliance with IEEE754)
	if (is_inf(x) || is_inf(y)) return INFINITY;

	// Order the operands
	auto ax = abs(x);
	auto ay = abs(y);
	if (ay > ax) { std::swap(ax, ay); }

	// Widely varying operands
	if (ay <= ax * ::cx::sqrt(std::numeric_limits<T>::epsilon() / T(2.0))) { return ax; }     // Note: This also gets ay == 0

	// Operands do not vary widely
	auto scale = eps(::cx::sqrt(std::numeric_limits<T>::min())); // Rescaling constant = 1.2924697f-26
	if (ax > ::cx::sqrt(std::numeric_limits<T>::max() / T(2.0))) {
		ax = ax * scale;
		ay = ay * scale;
		scale = T(1.0) / scale;
	}
	else if (ay < ::cx::sqrt(std::numeric_limits<T>::min())) {
		ax = ax / scale;
		ay = ay / scale;
	}
	else {
		scale = T(1.0);
	}

	auto h = sqrt(fma(ax, ax, ay * ay));
	// This branch is correctly rounded but requires a native hardware fma.
	const auto h_sq = h * h;
	const auto ax_sq = ax * ax;
	const auto cr = fma(-ay, ay, h_sq - ax_sq) + fma(h, h, -h_sq) - fma(ax, ax, -ax_sq);
	h = h - cr / (T(2.0) * h);
	return h * scale;
}

} // namespace detail

/// √(x²+y²)
constexpr float hypotf(float x, float y) noexcept { return detail::hypot_fma<float>(x,y); }
/// √(x²+y²)
constexpr float hypot(float x, float y) noexcept { return detail::hypot_fma<float>(x, y); }
/// √(x²+y²)
constexpr double hypot(double x, double y) noexcept { return detail::hypot_fma<double>(x,y); }

#if 0
/// √(x²+y²+z²)
constexpr double hypot(double x, double y, double z) noexcept {
    const auto maxabs = max(abs(x),abs(y),abs(z));
    if (isinf(maxabs) || is_zero(maxabs)) { return maxabs; }
    const auto rcp_maxabs = 1.0 / maxabs;
    return maxabs * sqrt(sum(sqr(x * rcp_maxabs),sqr(y * rcp_maxabs),sqr(z * rcp_maxabs)));
}
#else
/// √(x₁²+x₂²+ ... +xᵢ²)
template <typename... Ts> 
constexpr auto hypot(const Ts& ... x) /*noexcept*/ {
    const auto maxabs = max(abs(x)...);
	if (isinf(maxabs) || is_zero(maxabs)) { return maxabs; }
    using T = decltype(maxabs);
    const auto rcp_maxabs = T(1.0) / maxabs;
    return maxabs * sqrt(sum(sqr(x * rcp_maxabs)...));
    // return maxabs * sqrt(sum(sqr(x / maxabs)...));
}

// template <typename... Ts>
// constexpr auto test(const Ts& ... x) noexcept {
//     // return max(abs(x)...);
//     return sqrt(sum(sqr(x)...));
// }
#endif

/// OpenCl based functions 
#if 1
namespace ocl {


template<typename T>
constexpr T MATH_DIVIDE(T x, T y) { return (x / y); }

template<typename T>
constexpr T MATH_RECIP(T x) { return (1.0f / x); }

using uint = uint32_t;

// float
constexpr uint SIGNBIT_SP32      = 0x80000000;
constexpr uint EXSIGNBIT_SP32    = 0x7fffffff;
constexpr uint EXPBITS_SP32      = 0x7f800000;
constexpr uint MANTBITS_SP32     = 0x007fffff;
constexpr uint ONEEXPBITS_SP32   = 0x3f800000;
constexpr uint TWOEXPBITS_SP32   = 0x40000000;
constexpr uint HALFEXPBITS_SP32  = 0x3f000000;
constexpr uint IMPBIT_SP32       = 0x00800000;
constexpr uint QNANBITPATT_SP32  = 0x7fc00000;
constexpr uint INDEFBITPATT_SP32 = 0xffc00000;
constexpr uint PINFBITPATT_SP32  = 0x7f800000;
constexpr uint NINFBITPATT_SP32  = 0xff800000;
constexpr auto EXPBIAS_SP32      = 127		;
constexpr auto EXPSHIFTBITS_SP32 = 23		;
constexpr auto BIASEDEMIN_SP32   = 1		;
constexpr auto EMIN_SP32         = -126		;
constexpr auto BIASEDEMAX_SP32   = 254		;
constexpr auto EMAX_SP32         = 127		;
constexpr auto LAMBDA_SP32       = 1.0e30	;
constexpr auto MANTLENGTH_SP32   = 24		;
constexpr auto BASEDIGITS_SP32   = 7		;

/// double
constexpr auto SIGNBIT_DP64      = 0x8000000000000000L;
constexpr auto EXSIGNBIT_DP64    = 0x7fffffffffffffffL;
constexpr auto EXPBITS_DP64      = 0x7ff0000000000000L;
constexpr auto MANTBITS_DP64     = 0x000fffffffffffffL;
constexpr auto ONEEXPBITS_DP64   = 0x3ff0000000000000L;
constexpr auto TWOEXPBITS_DP64   = 0x4000000000000000L;
constexpr auto HALFEXPBITS_DP64  = 0x3fe0000000000000L;
constexpr auto IMPBIT_DP64       = 0x0010000000000000L;
constexpr auto QNANBITPATT_DP64  = 0x7ff8000000000000L;
constexpr auto INDEFBITPATT_DP64 = 0xfff8000000000000L;
constexpr auto PINFBITPATT_DP64  = 0x7ff0000000000000L;
constexpr auto NINFBITPATT_DP64  = 0xfff0000000000000L;
constexpr auto EXPBIAS_DP64      = 1023		;
constexpr auto EXPSHIFTBITS_DP64 = 52		;
constexpr auto BIASEDEMIN_DP64   = 1		;
constexpr auto EMIN_DP64         = -1022	;	
constexpr auto BIASEDEMAX_DP64   = 2046		; /* 0x7fe */
constexpr auto EMAX_DP64         = 1023		; /* 0x3ff */
constexpr auto LAMBDA_DP64       = 1.0e300	;
constexpr auto MANTLENGTH_DP64   = 53		;
constexpr auto BASEDIGITS_DP64   = 15		;

/// Reinterpret bits in a float as a signed integer.
__forceinline constexpr int32_t as_int(float f) { return std::bit_cast<int32_t>(f); }
/// Reinterpret bits in a double as a unsigned integer.
__forceinline constexpr int64_t as_int(double d) { return std::bit_cast<int64_t>(d); }

/// Reinterpret bits in a float as a unsigned integer.
__forceinline constexpr uint32_t as_uint(float f) { return std::bit_cast<uint32_t>(f); }

/// Reinterpret bits in a double as a unsigned integer.
__forceinline constexpr uint64_t as_uint(double d) { return std::bit_cast<uint64_t>(d); }
/// Reinterpret bits in a double as a unsigned integer.
__forceinline constexpr uint64_t as_ulong(double d) { return std::bit_cast<uint64_t>(d); }

/// Reinterpret bits in an integer as a float.
__forceinline constexpr float as_float(int32_t i) { return std::bit_cast<float>(i); }
/// Reinterpret bits in an unsigned integer as a float.
__forceinline constexpr float as_float(uint32_t u) { return std::bit_cast<float>(u); }

/// Reinterpret bits in an integer as a double. 
__forceinline constexpr double as_double(int64_t i) { return std::bit_cast<double>(i); }
/// Reinterpret bits in an unsigned integer as a double.
__forceinline constexpr double as_double(uint64_t u) { return std::bit_cast<double>(u); }

struct int2 {
	int lo;
	int hi;
};
/// Reinterpret bits in a double as a int2 struct.
__forceinline constexpr int2 as_int2(double d) { return std::bit_cast<int2>(d); }

// Approximates a * b + c. 
__forceinline constexpr float mad(float a, float b, float c) {
    // NOTE: GCC/ICC will turn this (for float) into a FMA unless explicitly asked not to, clang will do so if -ffp-contract=fast.
    CX_CLANG_PRAGMA(clang fp contract(fast))
    return a * b + c;
}

// Approximates a * b + c. 
__forceinline constexpr double mad(double a, double b, double c) {
    // NOTE: GCC/ICC will turn this (for float) into a FMA unless explicitly asked not to, clang will do so if -ffp-contract=fast.
    CX_CLANG_PRAGMA(clang fp contract(fast))
    return a * b + c;
}

/// acos
constexpr float acos(float x) noexcept {
    // Computes arccos(x).
    // The argument is first reduced by noting that arccos(x)
    // is invalid for abs(x) > 1. For denormal and small
    // arguments arccos(x) = pi/2 to machine accuracy.
    // Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arccos(x) = pi/2 - arcsin(x)
    // = pi/2 - (x + x^3*R(x^2))
    // where R(x^2) is a rational minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arccos(x) = pi - 2*arcsin(sqrt(1-x)/2)
    // together with the above rational approximation, and
    // reconstruct the terms carefully.


    // Some constants and split constants.
    const float piby2 = 1.5707963705e+00F;
    const float pi = 3.1415926535897933e+00F;
    const float piby2_head = 1.5707963267948965580e+00F;
    const float piby2_tail = 6.12323399573676603587e-17F;

    uint ux = as_uint(x);
    uint aux = ux & ~SIGNBIT_SP32;
    int xneg = ux != aux;
    int xexp = (int)(aux >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
    float y = as_float(aux);

    // transform if |x| >= 0.5
    int transform = xexp >= -1;

    float y2 = y * y;
    float yt = 0.5f * (1.0f - y);
    float r = transform ? yt : y2;

    // Use a rational approximation for [0.0, 0.5]
    float a = mad(r,
                  mad(r,
                      mad(r, -0.00396137437848476485201154797087F, -0.0133819288943925804214011424456F),
                      -0.0565298683201845211985026327361F),
                  0.184161606965100694821398249421F);

    float b = mad(r, -0.836411276854206731913362287293F, 1.10496961524520294485512696706F);
    float u = r * MATH_DIVIDE(a, b);

    float s = sqrtf(r);
    y = s;
    float s1 = as_float(as_uint(s) & 0xffff0000);
    float c = MATH_DIVIDE(mad(s1, -s1, r), s + s1);
    float rettn = mad(s + mad(y, u, -piby2_tail), -2.0f, pi);
    float rettp = 2.0F * (s1 + mad(y, u, c));
    float rett = xneg ? rettn : rettp;
    float ret = piby2_head - (x - mad(x, -u, piby2_tail));

    ret = transform ? rett : ret;
    ret = aux > 0x3f800000U ? as_float(QNANBITPATT_SP32) : ret;
    ret = ux == 0x3f800000U ? 0.0f : ret;
    ret = ux == 0xbf800000U ? pi : ret;
    ret = xexp < -26 ? piby2 : ret;
    return ret;
}

/// acos
constexpr double acos(double x) noexcept {
    // Computes arccos(x).
    // The argument is first reduced by noting that arccos(x)
    // is invalid for abs(x) > 1. For denormal and small
    // arguments arccos(x) = pi/2 to machine accuracy.
    // Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arccos(x) = pi/2 - arcsin(x)
    // = pi/2 - (x + x^3*R(x^2))
    // where R(x^2) is a rational minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arccos(x) = pi - 2*arcsin(sqrt(1-x)/2)
    // together with the above rational approximation, and
    // reconstruct the terms carefully.

    const double pi = 3.1415926535897933e+00;             /* 0x400921fb54442d18 */
    const double piby2 = 1.5707963267948965580e+00;       /* 0x3ff921fb54442d18 */
    const double piby2_head = 1.5707963267948965580e+00;  /* 0x3ff921fb54442d18 */
    const double piby2_tail = 6.12323399573676603587e-17; /* 0x3c91a62633145c07 */

    double y = fabs(x);
    int xneg = as_int2(x).hi < 0;
    int xexp = (as_int2(y).hi >> 20) - EXPBIAS_DP64;

    // abs(x) >= 0.5
    int transform = xexp >= -1;

    double rt = 0.5 * (1.0 - y);
    double y2 = y * y;
    double r = transform ? rt : y2;

    // Use a rational approximation for [0.0, 0.5]
    double un = fma(r,
                    fma(r,
                        fma(r,
                            fma(r,
                                fma(r, 0.0000482901920344786991880522822991,
                                       0.00109242697235074662306043804220),
                                -0.0549989809235685841612020091328),
                            0.275558175256937652532686256258),
                        -0.445017216867635649900123110649),
                    0.227485835556935010735943483075);

    double ud = fma(r,
                    fma(r,
                        fma(r,
                            fma(r, 0.105869422087204370341222318533,
                                   -0.943639137032492685763471240072),
                            2.76568859157270989520376345954),
                        -3.28431505720958658909889444194),
                    1.36491501334161032038194214209);

    double u = r * MATH_DIVIDE(un, ud);

    // Reconstruct acos carefully in transformed region
    double s = sqrt(r);
    double ztn =  fma(-2.0, (s + fma(s, u, -piby2_tail)), pi);

    double s1 = as_double(as_ulong(s) & 0xffffffff00000000UL);
    double c = MATH_DIVIDE(fma(-s1, s1, r), s + s1);
    double ztp = 2.0 * (s1 + fma(s, u, c));
    double zt =  xneg ? ztn : ztp;
    double z = piby2_head - (x - fma(-x, u, piby2_tail));

    z =  transform ? zt : z;

    z = xexp < -56 ? piby2 : z;
    z = isnan(x) ? as_double((as_ulong(x) | QNANBITPATT_DP64)) : z;
    z = x == 1.0 ? 0.0 : z;
    z = x == -1.0 ? pi : z;

    return z;
}

/// asin
constexpr float asin(float x) noexcept {
    // Computes arcsin(x).
    // The argument is first reduced by noting that arcsin(x)
    // is invalid for abs(x) > 1 and arcsin(-x) = -arcsin(x).
    // For denormal and small arguments arcsin(x) = x to machine
    // accuracy. Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arcsin(x) = x + x^3*R(x^2)
    // where R(x^2) is a rational minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arcsin(x) = pi/2 - 2*arcsin(sqrt(1-x)/2)
    // together with the above rational approximation, and
    // reconstruct the terms carefully.

    const float piby2_tail = 7.5497894159e-08F;   /* 0x33a22168 */
    const float hpiby2_head = 7.8539812565e-01F;  /* 0x3f490fda */
    const float piby2 = 1.5707963705e+00F;        /* 0x3fc90fdb */

    uint ux = as_uint(x);
    uint aux = ux & EXSIGNBIT_SP32;
    uint xs = ux ^ aux;
    float spiby2 = as_float(xs | as_uint(piby2));
    int xexp = (int)(aux >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
    float y = as_float(aux);

    // abs(x) >= 0.5
    int transform = xexp >= -1;

    float y2 = y * y;
    float rt = 0.5f * (1.0f - y);
    float r = transform ? rt : y2;

    // Use a rational approximation for [0.0, 0.5]
    float a = mad(r,
                  mad(r,
                      mad(r, -0.00396137437848476485201154797087F, -0.0133819288943925804214011424456F),
                      -0.0565298683201845211985026327361F),
                  0.184161606965100694821398249421F);

    float b = mad(r, -0.836411276854206731913362287293F, 1.10496961524520294485512696706F);
    float u = r * MATH_DIVIDE(a, b);

    float s = sqrtf(r);
    float s1 = as_float(as_uint(s) & 0xffff0000);
    float c = MATH_DIVIDE(mad(-s1, s1, r), s + s1);
    float p = mad(2.0f*s, u, -mad(c, -2.0f, piby2_tail));
    float q = mad(s1, -2.0f, hpiby2_head);
    float vt = hpiby2_head - (p - q);
    float v = mad(y, u, y);
    v = transform ? vt : v;

    float ret = as_float(xs | as_uint(v));
    ret = aux > 0x3f800000U ? as_float(QNANBITPATT_SP32) : ret;
    ret = aux == 0x3f800000U ? spiby2 : ret;
    ret = xexp < -14 ? x : ret;

    return ret;
}

/// asin
constexpr double asin(double x) noexcept {
    // Computes arcsin(x).
    // The argument is first reduced by noting that arcsin(x)
    // is invalid for abs(x) > 1 and arcsin(-x) = -arcsin(x).
    // For denormal and small arguments arcsin(x) = x to machine
    // accuracy. Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arcsin(x) = x + x^3*R(x^2)
    // where R(x^2) is a rational minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arcsin(x) = pi/2 - 2*arcsin(sqrt(1-x)/2)
    // together with the above rational approximation, and
    // reconstruct the terms carefully.

    const double piby2_tail = 6.1232339957367660e-17;  /* 0x3c91a62633145c07 */
    const double hpiby2_head = 7.8539816339744831e-01; /* 0x3fe921fb54442d18 */
    const double piby2 = 1.5707963267948965e+00;       /* 0x3ff921fb54442d18 */

    double y = fabs(x);
    int xneg = as_int2(x).hi < 0;
    int xexp = (as_int2(y).hi >> 20) - EXPBIAS_DP64;

    // abs(x) >= 0.5
    int transform = xexp >= -1;

    double rt = 0.5 * (1.0 - y);
    double y2 = y * y;
    double r = transform ? rt : y2;

    // Use a rational approximation for [0.0, 0.5]
    double un = fma(r,
                    fma(r,
                        fma(r,
                            fma(r,
                                fma(r, 0.0000482901920344786991880522822991,
                                       0.00109242697235074662306043804220),
                                -0.0549989809235685841612020091328),
                            0.275558175256937652532686256258),
                        -0.445017216867635649900123110649),
                    0.227485835556935010735943483075);

    double ud = fma(r,
                    fma(r,
                        fma(r,
                            fma(r, 0.105869422087204370341222318533,
                                   -0.943639137032492685763471240072),
                            2.76568859157270989520376345954),
                        -3.28431505720958658909889444194),
                    1.36491501334161032038194214209);

    double u = r * MATH_DIVIDE(un, ud);

    // Reconstruct asin carefully in transformed region
    double s = sqrt(r);
    double sh = as_double(as_ulong(s) & 0xffffffff00000000UL);
    double c = MATH_DIVIDE(fma(-sh, sh, r), s + sh);
    double p = fma(2.0*s, u, -fma(-2.0, c, piby2_tail));
    double q = fma(-2.0, sh, hpiby2_head);
    double vt = hpiby2_head - (p - q);
    double v = fma(y, u, y);
    v = transform ? vt : v;

    v = xexp < -28 ? y : v;
    v = xexp >= 0 ? as_double(QNANBITPATT_DP64) : v;
    v = y == 1.0 ? piby2 : v;

    return xneg ? -v : v;
}

/// atan
constexpr float atan(float x) noexcept {
    const float piby2 = 1.5707963267948966f; // 0x3ff921fb54442d18

    uint ux = as_uint(x);
    uint aux = ux & EXSIGNBIT_SP32;
    uint sx = ux ^ aux;

    float spiby2 = as_float(sx | as_uint(piby2));

    float v = as_float(aux);

    // Return for NaN
    float ret = x;

    // 2^26 <= |x| <= Inf => atan(x) is close to piby2
    ret = aux <= PINFBITPATT_SP32  ? spiby2 : ret;

    // Reduce arguments 2^-19 <= |x| < 2^26

    // 39/16 <= x < 2^26
    x = -MATH_RECIP(v);
    float c = 1.57079632679489655800f; // atan(infinity)

    // 19/16 <= x < 39/16
    int l = aux < 0x401c0000;
    float xx = MATH_DIVIDE(v - 1.5f, mad(v, 1.5f, 1.0f));
    x = l ? xx : x;
    c = l ? 9.82793723247329054082e-01f : c; // atan(1.5)

    // 11/16 <= x < 19/16
    l = aux < 0x3f980000U;
    xx =  MATH_DIVIDE(v - 1.0f, 1.0f + v);
    x = l ? xx : x;
    c = l ? 7.85398163397448278999e-01f : c; // atan(1)

    // 7/16 <= x < 11/16
    l = aux < 0x3f300000;
    xx = MATH_DIVIDE(mad(v, 2.0f, -1.0f), 2.0f + v);
    x = l ? xx : x;
    c = l ? 4.63647609000806093515e-01f : c; // atan(0.5)

    // 2^-19 <= x < 7/16
    l = aux < 0x3ee00000;
    x = l ? v : x;
    c = l ? 0.0f : c;

    // Core approximation: Remez(2,2) on [-7/16,7/16]

    float s = x * x;
    float a = mad(s,
                  mad(s, 0.470677934286149214138357545549e-2f, 0.192324546402108583211697690500f),
                  0.296528598819239217902158651186f);

    float b = mad(s,
                  mad(s, 0.299309699959659728404442796915f, 0.111072499995399550138837673349e1f),
                  0.889585796862432286486651434570f);

    float q = x * s * MATH_DIVIDE(a, b);

    float z = c - (q - x);
    float zs = as_float(sx | as_uint(z));

    ret  = aux < 0x4c800000 ?  zs : ret;

    // |x| < 2^-19
    ret = aux < 0x36000000 ? as_float(ux) : ret;
    return ret;
}

/// atan
//  time: 182 ms 
//  ulp:     1
constexpr double atan(double x) noexcept {
    const double piby2 = 1.5707963267948966e+00; // 0x3ff921fb54442d18

    double v = cx::fabs(x);

    // 2^56 > v > 39/16
    double a = -1.0;
    double b = v;
    // (chi + clo) = arctan(infinity)
    double chi = 1.57079632679489655800e+00;
    double clo = 6.12323399573676480327e-17;

    double ta = v - 1.5;
    double tb = 1.0 + 1.5 * v;
    int l = v <= 0x1.38p+1; // 39/16 > v > 19/16
    a = l ? ta : a; // select
    b = l ? tb : b; // select
    // (chi + clo) = arctan(1.5)
    chi = l ? 9.82793723247329054082e-01 : chi;
    clo = l ? 1.39033110312309953701e-17 : clo;

    ta = v - 1.0;
    tb = 1.0 + v;
    l = v <= 0x1.3p+0; // 19/16 > v > 11/16
    a = l ? ta : a; // select
    b = l ? tb : b; // select
    // (chi + clo) = arctan(1.)
    chi = l ? 7.85398163397448278999e-01 : chi;
    clo = l ? 3.06161699786838240164e-17 : clo;

    ta = 2.0 * v - 1.0;
    tb = 2.0 + v;
    l = v <= 0x1.6p-1; // 11/16 > v > 7/16
    a = l ? ta : a; // select
    b = l ? tb : b; // select
    // (chi + clo) = arctan(0.5)
    chi = l ? 4.63647609000806093515e-01 : chi;
    clo = l ? 2.26987774529616809294e-17 : clo;

    l = v <= 0x1.cp-2; // v < 7/16
    a = l ? v : a; // select
    b = l ? 1.0 : b; // select
    chi = l ? 0.0 : chi; // select
    clo = l ? 0.0 : clo; // select

    // Core approximation: Remez(4,4) on [-7/16,7/16]
    double r = a / b; // ! div
    const double s = r * r;
    double qn = fma(s,
                    fma(s,
                        fma(s,
                            fma(s, 0.142316903342317766e-3,
                                   0.304455919504853031e-1),
                            0.220638780716667420e0),
                        0.447677206805497472e0),
                    0.268297920532545909e0);

    double qd = fma(s,
	            fma(s,
			fma(s,
			    fma(s, 0.389525873944742195e-1,
				   0.424602594203847109e0),
                            0.141254259931958921e1),
                        0.182596787737507063e1),
                    0.804893761597637733e0);

    double q = r * s * qn / qd; // ! div
    r = chi - ((q - clo) - r);

    double z = piby2; //TODO: isnan(x) ? x : piby2;
    z = v <= 0x1.0p+56 ? r : z; // select
    z = v <  0x1.0p-26 ? v : z; // select
    return x == v ? z : -z; // select
}


} // namespace ocl
#endif


/// ISPC based functions (float only !)
#if 0
namespace ispc {

/// Reinterpret bits in an integer as a float. [ISPC]
__forceinline constexpr float floatbits(int32_t i) { return std::bit_cast<float>(i); }
/// Reinterpret bits in an unsigned integer as a float. [ISPC]
__forceinline constexpr float floatbits(uint32_t u) { return std::bit_cast<float>(u); }
/// Reinterpret bits in a 64-bit signed integer as a double. [ISPC]
__forceinline constexpr double doublebits(int64_t i) { return std::bit_cast<double>(i); }
/// Reinterpret bits in a 64-bit unsigned integer as a double. [ISPC]
__forceinline constexpr double doublebits(uint64_t u) { return std::bit_cast<double>(u); }
/// Reinterpret bits in a float as a unsigned integer. [ISPC]
__forceinline constexpr uint32_t intbits(float f) { return std::bit_cast<uint32_t>(f); }
/// Reinterpret bits in a double as a 64-bit unsigned integer. [ISPC]
__forceinline constexpr uint64_t intbits(double d) { return std::bit_cast<uint64_t>(d); }

// Most of the transcendental implementations in ispc code here come from
// Solomon Boulos's "syrah": https://github.com/boulos/syrah/

/// tan 
inline constexpr float tan(float x_full) {
    const float pi_over_four_vec = 0.785398185253143310546875f;
    const float four_over_pi_vec = 1.27323949337005615234375f;

    bool x_lt_0 = x_full < 0.;
    float y = x_lt_0 ? -x_full : x_full;
    float scaled = y * four_over_pi_vec;

    float k_real = floor(scaled);
    int k = (int)k_real;

    float x = y - k_real * pi_over_four_vec;

    // if k & 1, x -= Pi/4
    bool need_offset = (k & 1) != 0;
    x = need_offset ? x - pi_over_four_vec : x;

    // if k & 3 == (0 or 3) let z = tan_In...(y) otherwise z = -cot_In0To...
    int k_mod4 = k & 3;
    bool use_cotan = (k_mod4 == 1) || (k_mod4 == 2);

    const float one_vec = 1.0f;

    const float tan_c2 = 0.33333075046539306640625f;
    const float tan_c4 = 0.13339905440807342529296875f;
    const float tan_c6 = 5.3348250687122344970703125e-2f;
    const float tan_c8 = 2.46033705770969390869140625e-2f;
    const float tan_c10 = 2.892402000725269317626953125e-3f;
    const float tan_c12 = 9.500005282461643218994140625e-3f;

    const float cot_c2 = -0.3333333432674407958984375f;
    const float cot_c4 = -2.222204394638538360595703125e-2f;
    const float cot_c6 = -2.11752182804048061370849609375e-3f;
    const float cot_c8 = -2.0846328698098659515380859375e-4f;
    const float cot_c10 = -2.548247357481159269809722900390625e-5f;
    const float cot_c12 = -3.5257363606433500535786151885986328125e-7f;

    float x2 = x * x;
    float z;
    if (use_cotan) { // cif
        float cot_val = x2 * cot_c12 + cot_c10;
        cot_val = x2 * cot_val + cot_c8;
        cot_val = x2 * cot_val + cot_c6;
        cot_val = x2 * cot_val + cot_c4;
        cot_val = x2 * cot_val + cot_c2;
        cot_val = x2 * cot_val + one_vec;
        // The equation is for x * cot(x) but we need -x * cot(x) for the tan part.
        cot_val /= -x;
        z = cot_val;
    } else {
        float tan_val = x2 * tan_c12 + tan_c10;
        tan_val = x2 * tan_val + tan_c8;
        tan_val = x2 * tan_val + tan_c6;
        tan_val = x2 * tan_val + tan_c4;
        tan_val = x2 * tan_val + tan_c2;
        tan_val = x2 * tan_val + one_vec;
        // Equation was for tan(x)/x
        tan_val *= x;
        z = tan_val;
    }
    return x_lt_0 ? -z : z;
}

/// atan 
inline constexpr float atan(float x_full) {
    constexpr float pi_over_two_vec = 1.57079637050628662109375f;
	// atan(-x) = -atan(x) (so flip from negative to positive first)
	// if x > 1 -> atan(x) = Pi/2 - atan(1/x)
	bool x_neg = x_full < 0;
	float x_flipped = x_neg ? -x_full : x_full;

	bool x_gt_1 = x_flipped > 1.;
	float x = x_gt_1 ? 1.0f / x_flipped : x_flipped;

	// These coefficients approximate atan(x)/x
	const float atan_c0 = 0.99999988079071044921875f;
	const float atan_c2 = -0.3333191573619842529296875f;
	const float atan_c4 = 0.199689209461212158203125f;
	const float atan_c6 = -0.14015688002109527587890625f;
	const float atan_c8 = 9.905083477497100830078125e-2f;
	const float atan_c10 = -5.93664981424808502197265625e-2f;
	const float atan_c12 = 2.417283318936824798583984375e-2f;
	const float atan_c14 = -4.6721356920897960662841796875e-3f;

	float x2 = x * x;
	float result = x2 * atan_c14 + atan_c12;
	result = x2 * result + atan_c10;
	result = x2 * result + atan_c8;
	result = x2 * result + atan_c6;
	result = x2 * result + atan_c4;
	result = x2 * result + atan_c2;
	result = x2 * result + atan_c0;
	result *= x;

	result = x_gt_1 ? pi_over_two_vec - result : result;
	result = x_neg ? -result : result;
	return result;
}

/// atan2 
inline constexpr float atan2(float y, float x) {
    constexpr float pi_vec = 3.1415926536f;
	// atan2(y, x) =
	//
	// atan2(y > 0, x = +-0) ->  Pi/2
	// atan2(y < 0, x = +-0) -> -Pi/2
	// atan2(y = +-0, x < +0) -> +-Pi
	// atan2(y = +-0, x >= +0) -> +-0
	//
	// atan2(y >= 0, x < 0) ->  Pi + atan(y/x)
	// atan2(y <  0, x < 0) -> -Pi + atan(y/x)
	// atan2(y, x > 0) -> atan(y/x)
	//
	// and then a bunch of code for dealing with infinities.
	float y_over_x = y / x;
	float atan_arg = atan(y_over_x);
	bool x_lt_0 = x < 0.0f;
	bool y_lt_0 = y < 0.0f;
	float offset = x_lt_0 ? (y_lt_0 ? -pi_vec : pi_vec) : 0.0f;
	return offset + atan_arg;
}

} // namespace ispc
#endif

/// Enoki code
#if 1
namespace enoki {

namespace detail {

template <typename Value, size_t n>
__forceinline constexpr Value estrin_impl(const Value &x, const Value (&coeff)[n]) {
    constexpr size_t n_rec = (n - 1) / 2, n_fma = n / 2;

    Value coeff_rec[n_rec + 1];
    /*ENOKI_UNROLL*/ for (size_t i = 0; i < n_fma; ++i)
        coeff_rec[i] = fma(x, coeff[2 * i + 1], coeff[2 * i]);

    if constexpr (n_rec == n_fma) // odd case
        coeff_rec[n_rec] = coeff[n - 1];

    if constexpr (n_rec == 0)
        return coeff_rec[0];
    else
        return estrin_impl(sqr(x), coeff_rec);
}

template <typename Value, size_t n>
__forceinline constexpr Value horner_impl(const Value& x, const Value(&coeff)[n]) {
	Value accum = coeff[n - 1];
	/*ENOKI_UNROLL*/ for (size_t i = 1; i < n; ++i)
		accum = fma(x, accum, coeff[n - 1 - i]);
	return accum;
}

} // namespace detail

/// Estrin's scheme for polynomial evaluation
template <typename Value, typename... Ts>
__forceinline constexpr Value estrin(const Value& x, Ts... ts) {
	Value coeffs[]{ Value(ts)... }; // Value coeffs[]{ scalar_t<Value>(ts)... };
	return detail::estrin_impl(x, coeffs);
}

/// Horner's scheme for polynomial evaluation
template <typename Value, typename... Ts>
__forceinline constexpr Value horner(const Value& x, Ts... ts) {
	 Value coeffs[]{ Value(ts)... }; // Value coeffs[]{ scalar_t<Value>(ts)... };
	return detail::horner_impl(x, coeffs);
}

} // namespace enoki
#endif

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// SIN, COS, TAN, Trigonometric functions.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail { // vm::

// *************************************************************
//             sin/cos template, double precision
// *************************************************************
// Template parameters:
// VTYPE:  f.p. vector type
// SC:     1 = sin, 2 = cos, 3 = sincos
// Paramterers:
// xx = input x (radians)
// cosret = return pointer (only if SC = 3)
template<typename VTYPE, int SC>
static constexpr VTYPE sincos_d(VTYPE * cosret, VTYPE const xx) noexcept {

    // define constants
    const double P0sin = -1.66666666666666307295E-1;
    const double P1sin = 8.33333333332211858878E-3;
    const double P2sin = -1.98412698295895385996E-4;
    const double P3sin = 2.75573136213857245213E-6;
    const double P4sin = -2.50507477628578072866E-8;
    const double P5sin = 1.58962301576546568060E-10;

    const double P0cos = 4.16666666666665929218E-2;
    const double P1cos = -1.38888888888730564116E-3;
    const double P2cos = 2.48015872888517045348E-5;
    const double P3cos = -2.75573141792967388112E-7;
    const double P4cos = 2.08757008419747316778E-9;
    const double P5cos = -1.13585365213876817300E-11;

    const double DP1 = 7.853981554508209228515625E-1 * 2.;
    const double DP2 = 7.94662735614792836714E-9 * 2.;
    const double DP3 = 3.06161699786838294307E-17 * 2.;

#if 1 //TODO: proper detection 
    using ITYPE = int64_t;                      // integer vector type
    using UITYPE = uint64_t;                    // unsigned integer vector type
    using BVTYPE = decltype(xx < xx);           // boolean vector type
#else
    typedef decltype(roundi(xx)) ITYPE;          // integer vector type
    typedef decltype(nan_code(xx)) UITYPE;       // unsigned integer vector type
    typedef decltype(xx < xx) BVTYPE;            // boolean vector type
#endif

    VTYPE  xa, x, y, x2, s, c, sin1, cos1;       // data vectors
    ITYPE  q, signsin, signcos;              // integer vectors, 64 bit

    BVTYPE swap, overflow;                       // boolean vectors

    xa = abs(xx);

    // Find quadrant
    y = round(xa * (double)(2.0 / VM_PI));        // quadrant, as float
    q = roundi(y);                               // quadrant, as integer
    // Find quadrant
    //      0 -   pi/4 => 0
    //   pi/4 - 3*pi/4 => 1
    // 3*pi/4 - 5*pi/4 => 2
    // 5*pi/4 - 7*pi/4 => 3
    // 7*pi/4 - 8*pi/4 => 4

    // Reduce by extended precision modular arithmetic
    x = nmul_add(y, DP3, nmul_add(y, DP2, nmul_add(y, DP1, xa)));    // x = ((xa - y * DP1) - y * DP2) - y * DP3;

    // Expansion of sin and cos, valid for -pi/4 <= x <= pi/4
    x2 = x * x;
    s = polynomial_5(x2, P0sin, P1sin, P2sin, P3sin, P4sin, P5sin);
    c = polynomial_5(x2, P0cos, P1cos, P2cos, P3cos, P4cos, P5cos);
    s = mul_add(x * x2, s, x);                                       // s = x + (x * x2) * s;
    c = mul_add(x2 * x2, c, nmul_add(x2, 0.5, 1.0));                 // c = 1.0 - x2 * 0.5 + (x2 * x2) * c;

    // swap sin and cos if odd quadrant
    swap = BVTYPE((q & 1) != 0);

    // check for overflow
    overflow = BVTYPE(UITYPE(q) > 0x80000000000000);  // q big if overflow
    overflow &= is_finite(xa);
    s = select(overflow, 0.0, s);
    c = select(overflow, 1.0, c);

    if constexpr ((SC & 1) != 0) {  // calculate sin
        sin1 = select(swap, c, s);
        signsin = ((q << 62) ^ ITYPE(reinterpret_i(xx)));
        sin1 = sign_combine(sin1, reinterpret_d(signsin));
    }
    if constexpr ((SC & 2) != 0) {  // calculate cos
        cos1 = select(swap, s, c);
        signcos = ((q + 1) & 2) << 62;
        cos1 = reinterpret_d(reinterpret_i(cos1) ^ signcos);  //? cos1 ^= reinterpret_d(signcos);
    }
    if constexpr (SC == 3) {  // calculate both. cos returned through pointer
        *cosret = cos1;
    }
    if constexpr ((SC & 1) != 0) return sin1; else return cos1;
}

// *************************************************************
//             tan template, double precision
// *************************************************************
// Template parameters:
// VTYPE:  f.p. vector type
// Paramterers:
// x = input x (radians)
template<typename VTYPE>
static constexpr VTYPE tan_d(VTYPE const x) noexcept {

    // define constants
    const double DP1 = 7.853981554508209228515625E-1 * 2.;
    const double DP2 = 7.94662735614792836714E-9 * 2.;
    const double DP3 = 3.06161699786838294307E-17 * 2.;

    const double P2tan = -1.30936939181383777646E4;
    const double P1tan = 1.15351664838587416140E6;
    const double P0tan = -1.79565251976484877988E7;

    const double Q3tan = 1.36812963470692954678E4;
    const double Q2tan = -1.32089234440210967447E6;
    const double Q1tan = 2.50083801823357915839E7;
    const double Q0tan = -5.38695755929454629881E7;

    typedef decltype(x > x) BVTYPE;         // boolean vector type
#if 1 //TODO: proper detection 
    using UITYPE = uint64_t; //TODO: 
#else
    typedef decltype(nan_code(x)) UITYPE;   // unsigned integer vector type
#endif

    VTYPE  xa, y, z, zz, px, qx, tn, recip; // data vectors
    BVTYPE doinvert, xzero, overflow;       // boolean vectors

    xa = abs(x);

    // Find quadrant
    y = round(xa * (double)(2. / VM_PI));   // quadrant, as float
    auto q = roundi(y);                     // quadrant, as integer
    // Find quadrant
    //      0 -   pi/4 => 0
    //   pi/4 - 3*pi/4 => 1
    // 3*pi/4 - 5*pi/4 => 2
    // 5*pi/4 - 7*pi/4 => 3
    // 7*pi/4 - 8*pi/4 => 4

    // Reduce by extended precision modular arithmetic
    // z = ((xa - y * DP1) - y * DP2) - y * DP3;
    z = nmul_add(y, DP3, nmul_add(y, DP2, nmul_add(y, DP1, xa)));

    // Pade expansion of tan, valid for -pi/4 <= x <= pi/4
    zz = z * z;
    px = polynomial_2(zz, P0tan, P1tan, P2tan);
    qx = polynomial_4n(zz, Q0tan, Q1tan, Q2tan, Q3tan);

    // qx cannot be 0 for x <= pi/4
    tn = mul_add(px / qx, z * zz, z);            // tn = z + z * zz * px / qx;

    // if (q&2) tn = -1/tn
    doinvert = BVTYPE((q & 1) != 0);
    xzero = (xa == 0.);
    // avoid division by 0. We will not be using recip anyway if xa == 0.
    // tn never becomes exactly 0 when x = pi/2 so we only have to make
    // a special case for x == 0.
    recip = (-1.) / select(xzero, VTYPE(-1.), tn);
    tn = select(doinvert, recip, tn);
    tn = sign_combine(tn, x);       // get original sign

    overflow = BVTYPE(UITYPE(q) > 0x80000000000000) & is_finite(xa);
    tn = select(overflow, 0., tn);

    return tn;
}



// *************************************************************
//             sincos template, single precision
// *************************************************************
// Template parameters:
// VTYPE:  f.p. vector type
// SC:     1 = sin, 2 = cos, 3 = sincos, 4 = tan
// Paramterers:
// xx = input x (radians)
// cosret = return pointer (only if SC = 3)
template<typename VTYPE, int SC>
static constexpr VTYPE sincos_f(VTYPE * cosret, VTYPE const xx) noexcept {

    // define constants
    const float DP1F = 0.78515625f * 2.f;
    const float DP2F = 2.4187564849853515625E-4f * 2.f;
    const float DP3F = 3.77489497744594108E-8f * 2.f;

    const float P0sinf = -1.6666654611E-1f;
    const float P1sinf = 8.3321608736E-3f;
    const float P2sinf = -1.9515295891E-4f;

    const float P0cosf = 4.166664568298827E-2f;
    const float P1cosf = -1.388731625493765E-3f;
    const float P2cosf = 2.443315711809948E-5f;

#if 1 //TODO: proper detection 
	using ITYPE = int32_t;                      // integer vector type
	using UITYPE = uint32_t;                    // unsigned integer vector type
	using BVTYPE = decltype(xx < xx);           // boolean vector type
#else
    typedef decltype(roundi(xx)) ITYPE;          // integer vector type
    typedef decltype(nan_code(xx)) UITYPE;       // unsigned integer vector type
    typedef decltype(xx < xx) BVTYPE;            // boolean vector type
#endif

    VTYPE  xa, x, y, x2, s, c, sin1, cos1;       // data vectors
    ITYPE  q, signsin, signcos;                  // integer vectors
    BVTYPE swap, overflow;                       // boolean vectors

    xa = abs(xx);

    // Find quadrant
    y = round(xa * (float)(2.0 / VM_PI));         // quadrant, as float
    q = roundi(y);                               // quadrant, as integer
    //      0 -   pi/4 => 0
    //   pi/4 - 3*pi/4 => 1
    // 3*pi/4 - 5*pi/4 => 2
    // 5*pi/4 - 7*pi/4 => 3
    // 7*pi/4 - 8*pi/4 => 4

    // Reduce by extended precision modular arithmetic
    // x = ((xa - y * DP1F) - y * DP2F) - y * DP3F;
    x = nmul_add(y, DP3F, nmul_add(y, DP2F, nmul_add(y, DP1F, xa)));

    // A two-step reduction saves time at the cost of precision for very big x:
    //x = (xa - y * DP1F) - y * (DP2F+DP3F);

    // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
    x2 = x * x;
    s = polynomial_2(x2, P0sinf, P1sinf, P2sinf) * (x*x2) + x;
    c = polynomial_2(x2, P0cosf, P1cosf, P2cosf) * (x2*x2) + nmul_add(0.5f, x2, 1.0f);

    // swap sin and cos if odd quadrant
    swap = BVTYPE((q & 1) != 0);

    // check for overflow
    overflow = BVTYPE(UITYPE(q) > 0x2000000);  // q big if overflow
    overflow &= is_finite(xa);
    s = select(overflow, 0.0f, s);
    c = select(overflow, 1.0f, c);

    if constexpr ((SC & 5) != 0) {  // calculate sin
        sin1 = select(swap, c, s);
        signsin = ((q << 30) ^ ITYPE(reinterpret_i(xx)));
        sin1 = sign_combine(sin1, reinterpret_f(signsin));
    }
    if constexpr ((SC & 6) != 0) {  // calculate cos
        cos1 = select(swap, s, c);
        signcos = ((q + 1) & 2) << 30;
        cos1 = reinterpret_f(reinterpret_i(cos1) ^ signcos); // cos1 ^= reinterpret_f(signcos);
    }
    if constexpr (SC == 1) return sin1;
    else if constexpr (SC == 2) return cos1;
    else if constexpr (SC == 3) {  // calculate both. cos returned through pointer
        *cosret = cos1;
        return sin1;
    }
    else {  // SC == 4. tan
        return sin1 / cos1;
    }
}

} // namespace detail

/// sin(x)
constexpr float sin(float x) noexcept { return detail::sincos_f<float, 1>(0, x); }
/// sin(x)
constexpr float sinf(float x) noexcept { return detail::sincos_f<float, 1>(0, x); }
/// sin(x)
constexpr double sin(double x) noexcept { return detail::sincos_d<double, 1>(0, x); }

/// cos(x)
constexpr float cos(float x) noexcept { return detail::sincos_f<float, 2>(0, x); }
/// cos(x)
constexpr float cosf(float x) noexcept { return detail::sincos_f<float, 2>(0, x); }
/// cos(x)
constexpr double cos(double x) noexcept { return detail::sincos_d<double, 2>(0, x); }

//TODO: constexpr void sincos(float * cosret, float x) { }
//TODO: constexpr void sincos(double * cosret, double x) { }

/// tan(x)
constexpr float tan(float x) noexcept { return detail::sincos_f<float, 4>(0, x); }
/// tan(x)
constexpr float tanf(float x) noexcept { return detail::sincos_f<float, 4>(0, x); }
/// tan(x)
constexpr double tan(double x) noexcept { return detail::tan_d<double>(x); }

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// ASIN, ACOS, ATAN, Inverse trigonometric functions.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail { 

// *************************************************************
//             atan template, double precision
// *************************************************************
// Template parameters:
// VTYPE:  f.p. vector type
// T2:     0 = atan, 1 = atan2
// Paramterers:
// y, x. calculate tan(y/x)
// result is between -pi/2 and +pi/2 when x > 0
// result is between -pi and -pi/2 or between pi/2 and pi when x < 0 for atan2
template<typename VTYPE, int T2>
static constexpr VTYPE atan_d(VTYPE const y, VTYPE const x) noexcept {

    // define constants
    //const double ONEOPIO4 = 4./VM_PI;
    const double MOREBITS = 6.123233995736765886130E-17;
    const double MOREBITSO2 = MOREBITS * 0.5;
    const double T3PO8 = VM_SQRT2 + 1.; // 2.41421356237309504880;

    const double P4atan = -8.750608600031904122785E-1;
    const double P3atan = -1.615753718733365076637E1;
    const double P2atan = -7.500855792314704667340E1;
    const double P1atan = -1.228866684490136173410E2;
    const double P0atan = -6.485021904942025371773E1;

    const double Q4atan = 2.485846490142306297962E1;
    const double Q3atan = 1.650270098316988542046E2;
    const double Q2atan = 4.328810604912902668951E2;
    const double Q1atan = 4.853903996359136964868E2;
    const double Q0atan = 1.945506571482613964425E2;

    typedef decltype (x > x) BVTYPE;                            // boolean vector type
    VTYPE  t, x1, x2, y1, y2, s, fac, a, b, z, zz, px, qx, re;  // data vectors
    BVTYPE swapxy, notbig, notsmal;                             // boolean vectors

    if constexpr (T2 == 1) {  // atan2(y,x)
        // move in first octant
        x1 = abs(x);
        y1 = abs(y);
        swapxy = (y1 > x1);
        // swap x and y if y1 > x1
        x2 = select(swapxy, y1, x1);
        y2 = select(swapxy, x1, y1);

        // check for special case: x and y are both +/- INF
        BVTYPE both_infinite = is_inf(x) & is_inf(y);   // x and Y are both infinite
        if (horizontal_or(both_infinite)) [[unlikely]] {             // at least one element has both infinite
            VTYPE mone = VTYPE(-1.0);
            x2 = select(both_infinite, bit_and(x2, mone), x2);  // get 1.0 with the sign of x
            y2 = select(both_infinite, bit_and(y2, mone), y2);  // get 1.0 with the sign of y
        }

        if (std::is_constant_evaluated()) {
            if (x2 != 0.0) { t = y2 / x2; } else { t = 0.0; } //TODO: !
        } else {
            t = y2 / x2;                  // x = y = 0 gives NAN here
        }
    }
    else {    // atan(y)
        t = abs(y);
    }

    // small:  t < 0.66
    // medium: 0.66 <= t <= 2.4142 (1+sqrt(2))
    // big:    t > 2.4142
    notbig  = t <= T3PO8;  // t <= 2.4142
    notsmal = t >= 0.66;   // t >= 0.66

    s   = select(notbig, VTYPE(VM_PI_4), VTYPE(VM_PI_2));
    s   = bit_and(notsmal, s);                   // select(notsmal, s, 0.);
    fac = select(notbig, VTYPE(MOREBITSO2), VTYPE(MOREBITS));
    fac = bit_and(notsmal, fac);  //select(notsmal, fac, 0.);

    // small:  z = t / 1.0;
    // medium: z = (t-1.0) / (t+1.0);
    // big:    z = -1.0 / t;
    a = bit_and(notbig, t);                    // select(notbig, t, 0.);
    a = if_add(notsmal, a, -1.0);
    b = bit_and(notbig, VTYPE(1.0));            //  select(notbig, 1., 0.);
    b = if_add(notsmal, b, t);
    z = a / b;                         // division by 0 will not occur unless x and y are both 0

    zz = z * z;

    px = polynomial_4(zz, P0atan, P1atan, P2atan, P3atan, P4atan);
    qx = polynomial_5n(zz, Q0atan, Q1atan, Q2atan, Q3atan, Q4atan);

    re = mul_add(px / qx, z * zz, z);  // re = (px / qx) * (z * zz) + z;
    re += s + fac;

    if constexpr (T2 == 1) {           // atan2(y,x)
        // move back in place
        re = select(swapxy, VM_PI_2 - re, re);
        re = select(bit_or(x, y) == 0.0, 0.0, re);      // atan2(0,0) = 0 by convention
        re = select(sign_bit(x), VM_PI - re, re);// also for x = -0.
    }
    // get sign bit
    re = sign_combine(re, y);

    return re;
}

// *************************************************************
//             atan template, single precision
// *************************************************************
// Template parameters:
// VTYPE:  f.p. vector type
// T2:     0 = atan, 1 = atan2
// Paramterers:
// y, x. calculate tan(y/x)
// result is between -pi/2 and +pi/2 when x > 0
// result is between -pi and -pi/2 or between pi/2 and pi when x < 0 for atan2
template<typename VTYPE, int T2>
static constexpr VTYPE atan_f(VTYPE const y, VTYPE const x) noexcept {

	// define constants
	const float P3atanf = 8.05374449538E-2f;
	const float P2atanf = -1.38776856032E-1f;
	const float P1atanf = 1.99777106478E-1f;
	const float P0atanf = -3.33329491539E-1f;

	typedef decltype (x > x) BVTYPE;             // boolean vector type
	VTYPE  t, x1, x2, y1, y2, s, a, b, z, zz, re;// data vectors
	BVTYPE swapxy, notbig, notsmal;              // boolean vectors

	if constexpr (T2 == 1) {  // atan2(y,x)
		// move in first octant
		x1 = abs(x);
		y1 = abs(y);
		swapxy = (y1 > x1);
		// swap x and y if y1 > x1
		x2 = select(swapxy, y1, x1);
		y2 = select(swapxy, x1, y1);

		// check for special case: x and y are both +/- INF
		BVTYPE both_infinite = is_inf(x) & is_inf(y);   // x and Y are both infinite
		if (horizontal_or(both_infinite)) [[unlikely]] {             // at least one element has both infinite
			VTYPE mone = VTYPE(-1.0f);
			x2 = select(both_infinite, bit_and(x2, mone), x2);  // get 1.0 with the sign of x
			y2 = select(both_infinite, bit_and(y2, mone), y2);  // get 1.0 with the sign of y
		}

		// x = y = 0 will produce NAN. No problem, fixed below
		if (std::is_constant_evaluated()) {
            if (x2 != 0.0) { t = y2 / x2; } else { t = 0.0; } //TODO: !
		}
		else {
		    t = y2 / x2;
        }
	}
	else {    // atan(y)
		t = abs(y);
	}

	// small:  t < 0.4142
	// medium: 0.4142 <= t <= 2.4142
	// big:    t > 2.4142  (not for atan2)
	if constexpr (T2 == 0) {  // atan(y)
		notsmal = t >= float(VM_SQRT2 - 1.0);     // t >= tan  pi/8
		notbig = t <= float(VM_SQRT2 + 1.0);      // t <= tan 3pi/8

		s = select(notbig, VTYPE(float(VM_PI_4)), VTYPE(float(VM_PI_2)));
		s = bit_and(notsmal, s);                         // select(notsmal, s, 0.);

		// small:  z = t / 1.0;
		// medium: z = (t-1.0) / (t+1.0);
		// big:    z = -1.0 / t;
		a = bit_and(notbig, t);                // select(notbig, t, 0.);
		a = if_add(notsmal, a, -1.0f);
		b = bit_and(notbig, VTYPE(1.0f));       //  select(notbig, 1., 0.);
		b = if_add(notsmal, b, t);
		z = a / b;                     // division by 0 will not occur unless x and y are both 0
	}
	else {  // atan2(y,x)
		// small:  z = t / 1.0;
		// medium: z = (t-1.0) / (t+1.0);
		notsmal = t >= float(VM_SQRT2 - 1.0);
		a = if_add(notsmal, t, -1.0f);
		b = if_add(notsmal, 1.0f, t);
		s = bit_and(notsmal, VTYPE(float(VM_PI_4)));
		z = a / b;
	}

	zz = z * z;

	// Taylor expansion
	re = polynomial_3(zz, P0atanf, P1atanf, P2atanf, P3atanf);
	re = mul_add(re, zz * z, z) + s;

	if constexpr (T2 == 1) {                               // atan2(y,x)
		// move back in place
		re = select(swapxy, float(VM_PI_2) - re, re);
		re = select(bit_or(x, y) == 0.0f, 0.0f, re);       // atan2(0,+0) = 0 by convention
		re = select(sign_bit(x), float(VM_PI) - re, re);   // also for x = -0.
	}
	// get sign bit
	re = sign_combine(re, y);

	return re;
}


} // namespace detail

/// cos⁻¹(x)    ocl
constexpr float acos(float x) noexcept { return ocl::acos(x); }
/// cos⁻¹(x)    ocl
constexpr double acos(double x) noexcept { return ocl::acos(x); }

/// sin⁻¹(x)    ocl
constexpr float asin(float x) noexcept { return ocl::asin(x); }
/// sin⁻¹(x)    ocl
constexpr double asin(double x) noexcept { return ocl::asin(x); }

/// tan⁻¹(x)    ocl
constexpr float atan(float x) noexcept { return ocl::atan(x); } // slower as std::atan !
/// tan⁻¹(x)    ocl
constexpr double atan(double x) noexcept { return ocl::atan(x); } // slower as std::atan !

/// tan⁻¹(x)
constexpr float vm_atan(float y) noexcept { return detail::atan_f<float, 0>(y, 0.); } // slower as std::atan !
/// tan⁻¹(x)
constexpr double vm_atan(double y) noexcept { return detail::atan_d<double, 0>(y, 0.); } // slower as std::atan !

/// tan⁻¹(x/y)
constexpr float atan2(float y, float x) noexcept { return detail::atan_f<float, 1>(y, x); } //! WRONG 

/// tan⁻¹(x/y)
constexpr double atan2(double y, double x) noexcept { return detail::atan_d<double, 1>(y, x); } //! WRONG 


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// POW, LOG, EXP
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail { /// float pow(x,y)
/* ====================================================
* Copyright(C) 1993 by Sun Microsystems, Inc.All rights reserved.
*
* Developed at SunPro, a Sun Microsystems, Inc.business.
* Permission to use, copy, modify, and distribute this
* software is freely granted, provided that this notice
* is preserved.
* ====================================================*/

namespace float_pow_data {

static constexpr float
    bp[] = {1.0f, 1.5f,},
    dp_h[] = {0.0f, 5.84960938e-01f,}, /* 0x3f15c000 */
    dp_l[] = {0.0f, 1.56322085e-06f,}, /* 0x35d1cfdc */
    two24 = 16777216.0f,  /* 0x4b800000 */
    huge = 1.0e30f,
    tiny = 1.0e-30f,
    /* poly coefs for (3/2)*(log(x)-2s-2/3*s**3 */
	L1 = 6.0000002384e-01f, /* 0x3f19999a */
	L2 = 4.2857143283e-01f, /* 0x3edb6db7 */
	L3 = 3.3333334327e-01f, /* 0x3eaaaaab */
	L4 = 2.7272811532e-01f, /* 0x3e8ba305 */
	L5 = 2.3066075146e-01f, /* 0x3e6c3255 */
	L6 = 2.0697501302e-01f, /* 0x3e53f142 */
	P1 = 1.6666667163e-01f, /* 0x3e2aaaab */
    P2 = -2.7777778450e-03f, /* 0xbb360b61 */
    P3 = 6.6137559770e-05f, /* 0x388ab355 */
    P4 = -1.6533901999e-06f, /* 0xb5ddea0e */
    P5 = 4.1381369442e-08f, /* 0x3331bb4c */
    lg2 = 6.9314718246e-01f, /* 0x3f317218 */
    lg2_h = 6.93145752e-01f,   /* 0x3f317200 */
    lg2_l = 1.42860654e-06f,   /* 0x35bfbe8c */
    ovt = 4.2995665694e-08f, /* -(128-log2(ovfl+.5ulp)) */
    cp = 9.6179670095e-01f, /* 0x3f76384f =2/(3ln2) */
    cp_h = 9.6191406250e-01f, /* 0x3f764000 =12b cp */
    cp_l = -1.1736857402e-04f, /* 0xb8f623c6 =tail of cp_h */
    ivln2 = 1.4426950216e+00f, /* 0x3fb8aa3b =1/ln2 */
    ivln2_h = 1.4426879883e+00f, /* 0x3fb8aa00 =16b 1/ln2*/
    ivln2_l = 7.0526075433e-06f; /* 0x36eca570 =1/ln2 tail*/

}

/* Get a 32 bit int from a float.  */
__forceinline constexpr void GET_FLOAT_WORD(int32_t &w, float d) noexcept {                                           
	w = std::bit_cast<uint32_t>(d);
}

/* Set a float from a 32 bit int.  */
__forceinline constexpr void SET_FLOAT_WORD(float &d, int32_t w) noexcept {
	d = std::bit_cast<float>(w);
}

//=--------------------------------------------------------------------------------------------------------------------
constexpr float powf_impl(float x, float y) noexcept
{
	using namespace float_pow_data;

	float z,ax,z_h,z_l,p_h,p_l;
	float y1,t1,t2,r,s,sn,t,u,v,w;
	int32_t i,j,k,yisint,n;
	int32_t hx,hy,ix,iy,is;

	GET_FLOAT_WORD(hx, x);
	GET_FLOAT_WORD(hy, y);
	ix = hx & 0x7fffffff;
	iy = hy & 0x7fffffff;

	/* x**0 = 1, even if x is NaN */
	if (iy == 0)
		return 1.0f;
	/* 1**y = 1, even if y is NaN */
	if (hx == 0x3f800000)
		return 1.0f;
	/* NaN if either arg is NaN */
	if (ix > 0x7f800000 || iy > 0x7f800000)
		return x + y;

	/* determine if y is an odd int when x < 0
	 * yisint = 0       ... y is not an integer
	 * yisint = 1       ... y is an odd int
	 * yisint = 2       ... y is an even int
	 */
	yisint  = 0;
	if (hx < 0) {
		if (iy >= 0x4b800000)
			yisint = 2; /* even integer y */
		else if (iy >= 0x3f800000) {
			k = (iy>>23) - 0x7f;         /* exponent */
			j = iy>>(23-k);
			if ((j<<(23-k)) == iy)
				yisint = 2 - (j & 1);
		}
	}

	/* special value of y */
	if (iy == 0x7f800000) {  /* y is +-inf */
		if (ix == 0x3f800000)      /* (-1)**+-inf is 1 */
			return 1.0f;
		else if (ix > 0x3f800000)  /* (|x|>1)**+-inf = inf,0 */
			return hy >= 0 ? y : 0.0f;
		else if (ix != 0)          /* (|x|<1)**+-inf = 0,inf if x!=0 */
			return hy >= 0 ? 0.0f: -y;
	}
	if (iy == 0x3f800000)    /* y is +-1 */
		return hy >= 0 ? x : 1.0f/x;
	if (hy == 0x40000000)    /* y is 2 */
		return x*x;
	if (hy == 0x3f000000) {  /* y is  0.5 */
		if (hx >= 0)     /* x >= +0 */
			return cx::sqrtf(x);
	}

	ax = cx::fabsf(x);
	/* special value of x */
	if (ix == 0x7f800000 || ix == 0 || ix == 0x3f800000) { /* x is +-0,+-inf,+-1 */
		z = ax;
		if (hy < 0)  /* z = (1/|x|) */
			z = 1.0f/z;
		if (hx < 0) {
			if (((ix-0x3f800000)|yisint) == 0) {
				z = (z-z)/(z-z); /* (-1)**non-int is NaN */ //! warning C4723: potential divide by 0
			} else if (yisint == 1)
				z = -z;          /* (x<0)**odd = -(|x|**odd) */
		}
		return z;
	}

	sn = 1.0f; /* sign of result */
	if (hx < 0) {
		if (yisint == 0) /* (x<0)**(non-int) is NaN */
			return (x-x)/(x-x); //! warning C4723: potential divide by 0
		if (yisint == 1) /* (x<0)**(odd int) */
			sn = -1.0f;
	}

	/* |y| is huge */
	if (iy > 0x4d000000) { /* if |y| > 2**27 */
		/* over/underflow if x is not close to one */
		if (ix < 0x3f7ffff8)
			return hy < 0 ? sn*huge*huge : sn*tiny*tiny;
		if (ix > 0x3f800007)
			return hy > 0 ? sn*huge*huge : sn*tiny*tiny;
		/* now |1-x| is tiny <= 2**-20, suffice to compute
		   log(x) by x-x^2/2+x^3/3-x^4/4 */
		t = ax - 1;     /* t has 20 trailing zeros */
		w = (t*t)*(0.5f - t*(0.333333333333f - t*0.25f));
		u = ivln2_h*t;  /* ivln2_h has 16 sig. bits */
		v = t*ivln2_l - w*ivln2;
		t1 = u + v;
		GET_FLOAT_WORD(is, t1);
		SET_FLOAT_WORD(t1, is & 0xfffff000);
		t2 = v - (t1-u);
	} else {
		float s2,s_h,s_l,t_h,t_l;
		n = 0;
		/* take care subnormal number */
		if (ix < 0x00800000) {
			ax *= two24;
			n -= 24;
			GET_FLOAT_WORD(ix, ax);
		}
		n += ((ix)>>23) - 0x7f;
		j = ix & 0x007fffff;
		/* determine interval */
		ix = j | 0x3f800000;     /* normalize ix */
		if (j <= 0x1cc471)       /* |x|<sqrt(3/2) */
			k = 0;
		else if (j < 0x5db3d7)   /* |x|<sqrt(3)   */
			k = 1;
		else {
			k = 0;
			n += 1;
			ix -= 0x00800000;
		}
		SET_FLOAT_WORD(ax, ix);

		/* compute s = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5) */
		u = ax - bp[k];   /* bp[0]=1.0, bp[1]=1.5 */
		v = 1.0f/(ax+bp[k]);
		s = u*v;
		s_h = s;
		GET_FLOAT_WORD(is, s_h);
		SET_FLOAT_WORD(s_h, is & 0xfffff000);
		/* t_h=ax+bp[k] High */
		is = ((ix>>1) & 0xfffff000) | 0x20000000;
		SET_FLOAT_WORD(t_h, is + 0x00400000 + (k<<21));
		t_l = ax - (t_h - bp[k]);
		s_l = v*((u - s_h*t_h) - s_h*t_l);
		/* compute log(ax) */
		s2 = s*s;
		r = s2*s2*(L1+s2*(L2+s2*(L3+s2*(L4+s2*(L5+s2*L6)))));
		r += s_l*(s_h+s);
		s2 = s_h*s_h;
		t_h = 3.0f + s2 + r;
		GET_FLOAT_WORD(is, t_h);
		SET_FLOAT_WORD(t_h, is & 0xfffff000);
		t_l = r - ((t_h - 3.0f) - s2);
		/* u+v = s*(1+...) */
		u = s_h*t_h;
		v = s_l*t_h + t_l*s;
		/* 2/(3log2)*(s+...) */
		p_h = u + v;
		GET_FLOAT_WORD(is, p_h);
		SET_FLOAT_WORD(p_h, is & 0xfffff000);
		p_l = v - (p_h - u);
		z_h = cp_h*p_h;  /* cp_h+cp_l = 2/(3*log2) */
		z_l = cp_l*p_h + p_l*cp+dp_l[k];
		/* log2(ax) = (s+..)*2/(3*log2) = n + dp_h + z_h + z_l */
		t = (float)n;
		t1 = (((z_h + z_l) + dp_h[k]) + t);
		GET_FLOAT_WORD(is, t1);
		SET_FLOAT_WORD(t1, is & 0xfffff000);
		t2 = z_l - (((t1 - t) - dp_h[k]) - z_h);
	}

	/* split up y into y1+y2 and compute (y1+y2)*(t1+t2) */
	GET_FLOAT_WORD(is, y);
	SET_FLOAT_WORD(y1, is & 0xfffff000);
	p_l = (y-y1)*t1 + y*t2;
	p_h = y1*t1;
	z = p_l + p_h;
	GET_FLOAT_WORD(j, z);
	if (j > 0x43000000)          /* if z > 128 */
		return sn*huge*huge;  /* overflow */
	else if (j == 0x43000000) {  /* if z == 128 */
		if (p_l + ovt > z - p_h)
			return sn*huge*huge;  /* overflow */
	} else if ((j&0x7fffffff) > 0x43160000)  /* z < -150 */ // FIXME: check should be  (uint32_t)j > 0xc3160000
		return sn*tiny*tiny;  /* underflow */
	else if (std::bit_cast<uint32_t>(j) == 0xc3160000) {  /* z == -150 */ //TODO: warning: comparison of integer expressions of different signedness !
		if (p_l <= z-p_h)
			return sn*tiny*tiny;  /* underflow */
	}
	/*
	 * compute 2**(p_h+p_l)
	 */
	i = j & 0x7fffffff;
	k = (i>>23) - 0x7f;
	n = 0;
	if (i > 0x3f000000) {   /* if |z| > 0.5, set n = [z+0.5] */
		n = j + (0x00800000>>(k+1));
		k = ((n&0x7fffffff)>>23) - 0x7f;  /* new k for n */
		SET_FLOAT_WORD(t, n & ~(0x007fffff>>k));
		n = ((n&0x007fffff)|0x00800000)>>(23-k);
		if (j < 0)
			n = -n;
		p_h -= t;
	}
	t = p_l + p_h;
	GET_FLOAT_WORD(is, t);
	SET_FLOAT_WORD(t, is & 0xffff8000);
	u = t*lg2_h;
	v = (p_l-(t-p_h))*lg2 + t*lg2_l;
	z = u + v;
	w = v - (z - u);
	t = z*z;
	t1 = z - t*(P1+t*(P2+t*(P3+t*(P4+t*P5))));
	r = (z*t1)/(t1-2.0f) - (w+z*w);
	z = 1.0f - (r - z);
	GET_FLOAT_WORD(j, z);
	j += n<<23;
	if ((j>>23) <= 0)  /* subnormal output */
		z = cx::scalbnf(z, n);
	else
		SET_FLOAT_WORD(z, j);
	return sn*z;
}

} // namespace detail

namespace detail { /// double pow(x,y)



// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0f) = 0, exponent(0.0f) = -127, exponent(INF) = +128, exponent(NAN) = +128
static constexpr int32_t exponent(float a) noexcept {
	uint32_t t1 = std::bit_cast<uint32_t>(a);   // reinterpret as 32-bit integer
	uint32_t t2 = t1 << 1;               // shift out sign bit
	uint32_t t3 = t2 >> 24;              // shift down logical to position 0
    int32_t  t4 = int32_t(t3) - 0x7F;    // subtract bias from exponent
	return t4;
}


// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0) = 0, exponent(0.0) = -1023, exponent(INF) = +1024, exponent(NAN) = +1024
static constexpr int64_t exponent(double a) noexcept {
	uint64_t t1 = std::bit_cast<uint64_t>(a);   // reinterpret as 64-bit integer
	uint64_t t2 = t1 << 1;               // shift out sign bit
	uint64_t t3 = t2 >> 53;              // shift down logical to position 0
    int64_t  t4 = int64_t(t3) - 0x3FF;   // subtract bias from exponent
	return t4;
}

// extract exponent of a positive number x as a floating point number
static constexpr float exponent_f(float x) noexcept {
	const float pow2_23 = 8388608.0f;                   // 2^23
	const float bias = 127.f;                           // bias in exponent
	uint32_t a = std::bit_cast<uint32_t>(x);            // bit-cast x to integer
    uint32_t b = a >> 23;                               // shift down exponent to low bits
    uint32_t c = b | std::bit_cast<uint32_t>(pow2_23);  // insert new exponent
    float  d = std::bit_cast<float>(c);                 // bit-cast back to double
    float  e = d - (pow2_23 + bias);                    // subtract magic number and bias
	return e;
}

// extract exponent of a positive number x as a floating point number
static constexpr double exponent_f(double x) noexcept {
	const double pow2_52 = 4503599627370496.0;          // 2^52
	const double bias = 1023.0;                         // bias in exponent
	uint64_t a = std::bit_cast<uint64_t>(x);            // bit-cast x to integer
    uint64_t b = a >> 52;                               // shift down exponent to low bits
    uint64_t c = b | std::bit_cast<uint64_t>(pow2_52);  // insert new exponent
    double  d = std::bit_cast<double>(c);               // bit-cast back to double
    double  e = d - (pow2_52 + bias);                   // subtract magic number and bias
	return e;
}


static constexpr float fraction_2(float a) noexcept {
	auto t1 = std::bit_cast<uint32_t>(a);                  // reinterpret as 32-bit integer
	auto t2 = uint32_t((t1 & 0x007FFFFF) | 0x3F000000);    // set exponent to 0 + bias
	return std::bit_cast<float>(t2);
}

static constexpr double fraction_2(double a) noexcept {
	auto t1 = std::bit_cast<uint64_t>(a);                                   // reinterpret as 64-bit integer
	auto t2 = uint64_t((t1 & 0x000FFFFFFFFFFFFFll) | 0x3FE0000000000000ll); // set exponent to 0 + bias
	return std::bit_cast<double>(t2);
}

#if 1 

// Multiply and subtract with extra precision on the intermediate calculations,
// even if FMA instructions not supported, using Veltkamp-Dekker split.
// This is used in mathematical functions. Do not use it in general code
// because it is inaccurate in certain cases
template <typename VTYPE>
static constexpr VTYPE mul_sub_x(VTYPE a, VTYPE b, VTYPE c) noexcept {
	// return _mm_fmsub_ps(a, b, c);
    typedef decltype(roundi(a)) ITYPE;          // integer vector type

	// calculate a * b - c with extra precision
    ITYPE upper_mask = -(1 << 12);                         // mask to remove lower 12 bits
	VTYPE a_high = std::bit_cast<VTYPE>(std::bit_cast<ITYPE>(a) & upper_mask);// split into high and low parts
	VTYPE b_high = std::bit_cast<VTYPE>(std::bit_cast<ITYPE>(b) & upper_mask);
	VTYPE a_low = a - a_high;
	VTYPE b_low = b - b_high;
	VTYPE r1 = a_high * b_high;                            // this product is exact
	VTYPE r2 = r1 - c;                                     // subtract c from high product
	VTYPE r3 = r2 + (a_high * b_low + b_high * a_low) + a_low * b_low; // add rest of product
	return r3; // + ((r2 - r1) + c);
}

static constexpr double wm_pow_case_x0(bool const xiszero, double const y, double const z) noexcept {
	return select(xiszero, select(y < 0., infinite_vec<double>(), select(y == 0., double(1.), double(0.))), z);
}

static constexpr float wm_pow_case_x0(bool const xiszero, float const y, float const z) noexcept {
	return select(xiszero, select(y < 0.f, infinite_vec<float>(), select(y == 0.f, float(1.f), float(0.f))), z);
}

// ****************************************************************************
//                pow template, double precision
// ****************************************************************************
// Calculate x to the power of y.

// Precision is important here because rounding errors get multiplied by y.
// The logarithm is calculated with extra precision, and the exponent is
// calculated separately.
// The logarithm is calculated by Pade approximation with 6'th degree
// polynomials. A 7'th degree would be preferred for best precision by high y.
// The alternative method: log(x) = z + z^3*R(z)/S(z), where z = 2(x-1)/(x+1)
// did not give better precision.

// Template parameters:
// VTYPE:  data vector type
template <typename VTYPE>
static constexpr VTYPE pow_template_d(VTYPE const x0, VTYPE const y) noexcept {

    // define constants
    const double ln2d_hi = 0.693145751953125;           // log(2) in extra precision, high bits
    const double ln2d_lo = 1.42860682030941723212E-6;   // low bits of log(2)
    const double log2e   = VM_LOG2E;                    // 1/log(2)

    // coefficients for Pade polynomials
    const double P0logl =  2.0039553499201281259648E1;
    const double P1logl =  5.7112963590585538103336E1;
    const double P2logl =  6.0949667980987787057556E1;
    const double P3logl =  2.9911919328553073277375E1;
    const double P4logl =  6.5787325942061044846969E0;
    const double P5logl =  4.9854102823193375972212E-1;
    const double P6logl =  4.5270000862445199635215E-5;
    const double Q0logl =  6.0118660497603843919306E1;
    const double Q1logl =  2.1642788614495947685003E2;
    const double Q2logl =  3.0909872225312059774938E2;
    const double Q3logl =  2.2176239823732856465394E2;
    const double Q4logl =  8.3047565967967209469434E1;
    const double Q5logl =  1.5062909083469192043167E1;

    // Taylor coefficients for exp function, 1/n!
    const double p2  = 1./2.;
    const double p3  = 1./6.;
    const double p4  = 1./24.;
    const double p5  = 1./120.;
    const double p6  = 1./720.;
    const double p7  = 1./5040.;
    const double p8  = 1./40320.;
    const double p9  = 1./362880.;
    const double p10 = 1./3628800.;
    const double p11 = 1./39916800.;
    const double p12 = 1./479001600.;
    const double p13 = 1./6227020800.;

    typedef decltype(roundi(x0)) ITYPE;          // integer vector type
    typedef decltype(x0 < x0) BVTYPE;            // boolean vector type

    // data vectors
    VTYPE x, x1, x2;                             // x variable
    VTYPE px, qx, ef, yr, v;                     // calculation of logarithm
    VTYPE lg, lg1;
    VTYPE lgerr, x2err;
    VTYPE e1, e2, ee;
    VTYPE e3, z, z1;                             // calculation of exp and pow
    VTYPE yodd(0.0);                               // has sign bit set if y is an odd integer
    // integer vectors
    ITYPE ei, ej;
    // boolean vectors
    BVTYPE blend, xzero, xsign;                  // x conditions
    BVTYPE overflow, underflow, xfinite, yfinite, efinite; // error conditions

    // remove sign
    x1 = abs(x0);

    // Separate mantissa from exponent
    // This gives the mantissa * 0.5
    x  = fraction_2(x1);

    // reduce range of x = +/- sqrt(2)/2
    blend = x > VM_SQRT2*0.5;
    x  = if_add(!blend, x, x);                   // conditional add

    // Pade approximation
    // Higher precision than in log function. Still higher precision wanted
    x -= 1.0;
    x2 = x*x;
    px = polynomial_6  (x, P0logl, P1logl, P2logl, P3logl, P4logl, P5logl, P6logl);
    px *= x * x2;
    qx = polynomial_6n (x, Q0logl, Q1logl, Q2logl, Q3logl, Q4logl, Q5logl);
    lg1 = px / qx;

    // extract exponent
    ef = exponent_f(x1);
    ef = if_add(blend, ef, 1.);                  // conditional add

    // multiply exponent by y
    // nearest integer e1 goes into exponent of result, remainder yr is added to log
    e1 = round(ef * y);
    yr = mul_sub_x(ef, y, e1);                   // calculate remainder yr. precision very important here

    // add initial terms to Pade expansion
    lg = nmul_add(0.5, x2, x) + lg1;             // lg = (x - 0.5 * x2) + lg1;
    // calculate rounding errors in lg
    // rounding error in multiplication 0.5*x*x
    x2err = mul_sub_x(0.5*x, x, 0.5*x2);
    // rounding error in additions and subtractions
    lgerr = mul_add(0.5, x2, lg - x) - lg1;      // lgerr = ((lg - x) + 0.5 * x2) - lg1;

    // extract something for the exponent
    e2 = round(lg * y * VM_LOG2E);
    // subtract this from lg, with extra precision
    v = mul_sub_x(lg, y, e2 * ln2d_hi);
    v = nmul_add(e2, ln2d_lo, v);                // v -= e2 * ln2d_lo;

    // add remainder from ef * y
    v = mul_add(yr, VM_LN2, v);                  // v += yr * VM_LN2;

    // correct for previous rounding errors
    v = nmul_add(lgerr + x2err, y, v);           // v -= (lgerr + x2err) * y;

    // exp function

    // extract something for the exponent if possible
    x = v;
    e3 = round(x*log2e);
    // high precision multiplication not needed here because abs(e3) <= 1
    x = nmul_add(e3, VM_LN2, x);                 // x -= e3 * VM_LN2;

    z = polynomial_13m(x, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
    z = z + 1.0;

    // contributions to exponent
    ee = e1 + e2 + e3;
    //ei = round_to_int64_limited(ee);
    ei = roundi(ee);
    // biased exponent of result:
    ej = ei + (ITYPE(reinterpret_i(z)) >> 52);
    // check exponent for overflow and underflow
    overflow  = BVTYPE(ej >= 0x07FF) | (ee >  3000.);
    underflow = BVTYPE(ej <= 0x0000) | (ee < -3000.);

    // add exponent by integer addition
    //  z = reinterpret_d(ITYPE(reinterpret_i(z)) + (ei << 52));
    z = reinterpret_d(ITYPE(reinterpret_i(z)) + (reinterpret_u(ei) << 52));

    // check for special cases
    xfinite   = is_finite(x0);
    yfinite   = is_finite(y);
    efinite   = is_finite(ee);
    xzero     = is_zero_or_subnormal(x0);
    xsign     = sign_bit(x0);  // sign of x0. include -0.

    //printf(" [%i %f]  ",xsign?1:0,x0);

    // check for overflow and underflow
    if (horizontal_or(overflow | underflow)) {
        // handle errors
        z = select(underflow, VTYPE(0.), z);
        z = select(overflow, infinite_vec<VTYPE>(), z);
    }
    
    // check for x == 0
    z = wm_pow_case_x0(xzero, y, z);
    //z = select(xzero, select(y < 0., infinite_vec<VTYPE>(), select(y == 0., VTYPE(1.), VTYPE(0.))), z);

    // check for sign of x (include -0.). y must be integer
    if (horizontal_or(xsign)) {
        // test if y is an integer
        BVTYPE yinteger = (y == round(y));

        // test if y is odd: convert to int and shift bit 0 into position of sign bit.
        // this will be 0 if overflow
        //  yodd = reinterpret_d(roundi(y) << 63); //? possible UB ?
        yodd = reinterpret_d(reinterpret_u(roundi(y)) << 63);

        z1 = select(yinteger, bit_or(z, yodd),                    // y is integer. get sign if y is odd
            select(x0 == 0.0, z, nan_vec<VTYPE>(NAN_POW))); // NAN unless x0 == -0.
        yodd = select(yinteger, yodd, 0.0);                 // yodd used below. only if y is integer
        z = select(xsign, z1, z);
    }

    // check for range errors
    if (horizontal_and(xfinite & yfinite & (efinite | xzero))) {
        // fast return if no special cases
        return z;
    }

    // handle special error cases: y infinite
    z1 = select(yfinite & efinite, z,
        select(x1 == 1., VTYPE(1.),
            select((x1 > 1.) ^ sign_bit(y), infinite_vec<VTYPE>(), 0.)));

    // handle x infinite
    z1 = select(xfinite, z1,
        select(y == 0., VTYPE(1.),
            select(y < 0., bit_and(yodd, z),      // 0.0 with the sign of z from above
                bit_or(abs(x0), bit_and(x0, yodd))))); // get sign of x0 only if y is odd integer

    // Always propagate nan:
    // Deliberately differing from the IEEE-754 standard which has pow(0,nan)=1, and pow(1,nan)=1
    z1 = select(is_nan(x0)|is_nan(y), x0+y, z1);

    return z1;
}


// ****************************************************************************
//                pow template, single precision
// ****************************************************************************

// Template parameters:
// VTYPE:  data vector type
// Calculate x to the power of y
template <typename VTYPE>
static constexpr VTYPE pow_template_f(VTYPE const x0, VTYPE const y) noexcept {

	// define constants
	const float ln2f_hi = 0.693359375f;        // log(2), split in two for extended precision
	const float ln2f_lo = -2.12194440e-4f;
	const float log2e = float(VM_LOG2E);     // 1/log(2)

	const float P0logf = 3.3333331174E-1f;     // coefficients for logarithm expansion
	const float P1logf = -2.4999993993E-1f;
	const float P2logf = 2.0000714765E-1f;
	const float P3logf = -1.6668057665E-1f;
	const float P4logf = 1.4249322787E-1f;
	const float P5logf = -1.2420140846E-1f;
	const float P6logf = 1.1676998740E-1f;
	const float P7logf = -1.1514610310E-1f;
	const float P8logf = 7.0376836292E-2f;

	const float p2expf = 1.f / 2.f;             // coefficients for Taylor expansion of exp
	const float p3expf = 1.f / 6.f;
	const float p4expf = 1.f / 24.f;
	const float p5expf = 1.f / 120.f;
	const float p6expf = 1.f / 720.f;
	const float p7expf = 1.f / 5040.f;

	typedef decltype(roundi(x0)) ITYPE;          // integer vector type
	typedef decltype(x0 < x0) BVTYPE;            // boolean vector type

												 // data vectors
	VTYPE x, x1, x2;                             // x variable
	VTYPE ef, e1, e2, e3, ee;                    // exponent
	VTYPE yr;                                    // remainder
	VTYPE lg, lg1, lgerr, x2err, v;              // logarithm
	VTYPE z, z1;                                 // pow(x,y)
	VTYPE yodd(0);                               // has sign bit set if y is an odd integer
												 // integer vectors
	ITYPE ei, ej;                                // exponent
												 // boolean vectors
	BVTYPE blend, xzero, xsign;                  // x conditions
	BVTYPE overflow, underflow, xfinite, yfinite, efinite; // error conditions

														   // remove sign
	x1 = abs(x0);

	// Separate mantissa from exponent
	// This gives the mantissa * 0.5
	x = fraction_2(x1);

	// reduce range of x = +/- sqrt(2)/2
	blend = x > float(VM_SQRT2 * 0.5);
	x = if_add(!blend, x, x);                   // conditional add

												 // Taylor expansion, high precision
	x -= 1.0f;
	x2 = x * x;
	lg1 = polynomial_8(x, P0logf, P1logf, P2logf, P3logf, P4logf, P5logf, P6logf, P7logf, P8logf);
	lg1 *= x2 * x;

	// extract exponent
	ef = exponent_f(x1);
	ef = if_add(blend, ef, 1.0f);                // conditional add

												 // multiply exponent by y
												 // nearest integer e1 goes into exponent of result, remainder yr is added to log
	e1 = round(ef * y);
	yr = mul_sub_x(ef, y, e1);                   // calculate remainder yr. precision very important here

												 // add initial terms to expansion
	lg = nmul_add(0.5f, x2, x) + lg1;            // lg = (x - 0.5f * x2) + lg1;

												 // calculate rounding errors in lg
												 // rounding error in multiplication 0.5*x*x
	x2err = mul_sub_x(0.5f * x, x, 0.5f * x2);
	// rounding error in additions and subtractions
	lgerr = mul_add(0.5f, x2, lg - x) - lg1;     // lgerr = ((lg - x) + 0.5f * x2) - lg1;

												 // extract something for the exponent
	e2 = round(lg * y * float(VM_LOG2E));
	// subtract this from lg, with extra precision
	v = mul_sub_x(lg, y, e2 * ln2f_hi);
	v = nmul_add(e2, ln2f_lo, v);                // v -= e2 * ln2f_lo;

												 // correct for previous rounding errors
	v -= mul_sub(lgerr + x2err, y, yr * float(VM_LN2)); // v -= (lgerr + x2err) * y - yr * float(VM_LN2) ;

	// exp function

	// extract something for the exponent if possible
	x = v;
	e3 = round(x * log2e);
	// high precision multiplication not needed here because abs(e3) <= 1
	x = nmul_add(e3, float(VM_LN2), x);          // x -= e3 * float(VM_LN2);

	// Taylor polynomial
	x2 = x * x;
	z = polynomial_5(x, p2expf, p3expf, p4expf, p5expf, p6expf, p7expf) * x2 + x + 1.0f;

	// contributions to exponent
	ee = e1 + e2 + e3;
	ei = roundi(ee);
	// biased exponent of result:
	ej = ei + (ITYPE(reinterpret_i(z)) >> 23);
	// check exponent for overflow and underflow
	overflow = BVTYPE(ej >= 0x0FF) | (ee > 300.f);
	underflow = BVTYPE(ej <= 0x000) | (ee < -300.f);

	// add exponent by integer addition
	z = reinterpret_f(ITYPE(reinterpret_i(z)) + (ei << 23)); // the extra 0x10000 is shifted out here

															 // check for special cases
	xfinite = is_finite(x0);
	yfinite = is_finite(y);
	efinite = is_finite(ee);

	xzero = is_zero_or_subnormal(x0);
	xsign = sign_bit(x0);  // x is negative or -0.

							   // check for overflow and underflow
	if (horizontal_or(overflow | underflow)) {
		// handle errors
		z = select(underflow, VTYPE(0.f), z);
		z = select(overflow, infinite_vec<VTYPE>(), z);
	}

	// check for x == 0
	z = wm_pow_case_x0(xzero, y, z);
	//z = select(xzero, select(y < 0.f, infinite_vec<VTYPE>(), select(y == 0.f, VTYPE(1.f), VTYPE(0.f))), z);

	// check for sign of x (include -0.). y must be integer
	if (horizontal_or(xsign)) {
		// test if y is an integer
		BVTYPE yinteger = y == round(y);
		// test if y is odd: convert to int and shift bit 0 into position of sign bit.
		// this will be 0 if overflow
		yodd = reinterpret_f(roundi(y) << 31);
		z1 = select(yinteger, bit_or(z, yodd),                    // y is integer. get sign if y is odd
			select(x0 == 0.f, z, nan_vec<VTYPE>(NAN_POW)));// NAN unless x0 == -0.
		yodd = select(yinteger, yodd, VTYPE(0));                  // yodd used below. only if y is integer
		z = select(xsign, z1, z);
	}

	// check for range errors
	if (horizontal_and(xfinite & yfinite & (efinite | xzero))) {
		return z;            // fast return if no special cases
	}

	// handle special error cases: y infinite
	z1 = select(yfinite & efinite, z,
		select(x1 == 1.f, VTYPE(1.f),
			select((x1 > 1.f) ^ sign_bit(y), infinite_vec<VTYPE>(), 0.f)));

	// handle x infinite
	z1 = select(xfinite, z1,
		select(y == 0.f, VTYPE(1.f),
			select(y < 0.f, bit_and(yodd, z),     // 0.0 with the sign of z from above
                bit_or(abs(x0), bit_and(x0, yodd))))); // get sign of x0 only if y is odd integer

										  // Always propagate nan:
										  // Deliberately differing from the IEEE-754 standard which has pow(0,nan)=1, and pow(1,nan)=1
	z1 = select(is_nan(x0) | is_nan(y), x0 + y, z1);
	return z1;
}


#endif

} // namespace detail

namespace detail { /// double pow(x,y)

/* ====================================================
 * Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ==================================================== */
/* pow(x,y) return x**y
 *
 *                    n
 * Method:  Let x =  2   * (1+f)
 *      1. Compute and return log2(x) in two pieces:
 *              log2(x) = w1 + w2,
 *         where w1 has 53-24 = 29 bit trailing zeros.
 *      2. Perform y*log2(x) = n+y' by simulating muti-precision
 *         arithmetic, where |y'|<=0.5.
 *      3. Return x**y = 2**n*exp(y'*log2)
 *
 * Special cases:
 *      1.  (anything) ** 0  is 1
 *      2.  1 ** (anything)  is 1
 *      3.  (anything except 1) ** NAN is NAN
 *      4.  NAN ** (anything except 0) is NAN
 *      5.  +-(|x| > 1) **  +INF is +INF
 *      6.  +-(|x| > 1) **  -INF is +0
 *      7.  +-(|x| < 1) **  +INF is +0
 *      8.  +-(|x| < 1) **  -INF is +INF
 *      9.  -1          ** +-INF is 1
 *      10. +0 ** (+anything except 0, NAN)               is +0
 *      11. -0 ** (+anything except 0, NAN, odd integer)  is +0
 *      12. +0 ** (-anything except 0, NAN)               is +INF, raise divbyzero
 *      13. -0 ** (-anything except 0, NAN, odd integer)  is +INF, raise divbyzero
 *      14. -0 ** (+odd integer) is -0
 *      15. -0 ** (-odd integer) is -INF, raise divbyzero
 *      16. +INF ** (+anything except 0,NAN) is +INF
 *      17. +INF ** (-anything except 0,NAN) is +0
 *      18. -INF ** (+odd integer) is -INF
 *      19. -INF ** (anything) = -0 ** (-anything), (anything except odd integer)
 *      20. (anything) ** 1 is (anything)
 *      21. (anything) ** -1 is 1/(anything)
 *      22. (-anything) ** (integer) is (-1)**(integer)*(+anything**integer)
 *      23. (-anything except 0 and inf) ** (non-integer) is NAN
 *
 * Accuracy:
 *      pow(x,y) returns x**y nearly rounded. In particular
 *                      pow(integer,integer)
 *      always returns the correct integer provided it is
 *      representable.
 *
 * Constants :
 * The hexadecimal values are the intended ones for the following
 * constants. The decimal values may be used, provided that the
 * compiler will convert from decimal to binary accurately enough
 * to produce the hexadecimal values shown.
 */

namespace double_pow_data {

static constexpr double
    bp[]   = {1.0, 1.5,},
    dp_h[] = { 0.0, 5.84962487220764160156e-01,}, /* 0x3FE2B803, 0x40000000 */
    dp_l[] = { 0.0, 1.35003920212974897128e-08,}, /* 0x3E4CFDEB, 0x43CFD006 */
    two53  =  9007199254740992.0, /* 0x43400000, 0x00000000 */
    huge   =  1.0e300,
    tiny   =  1.0e-300,
    /* poly coefs for (3/2)*(log(x)-2s-2/3*s**3 */
    L1 =  5.99999999999994648725e-01, /* 0x3FE33333, 0x33333303 */
    L2 =  4.28571428578550184252e-01, /* 0x3FDB6DB6, 0xDB6FABFF */
    L3 =  3.33333329818377432918e-01, /* 0x3FD55555, 0x518F264D */
    L4 =  2.72728123808534006489e-01, /* 0x3FD17460, 0xA91D4101 */
    L5 =  2.30660745775561754067e-01, /* 0x3FCD864A, 0x93C9DB65 */
    L6 =  2.06975017800338417784e-01, /* 0x3FCA7E28, 0x4A454EEF */
    P1 =  1.66666666666666019037e-01, /* 0x3FC55555, 0x5555553E */
    P2 = -2.77777777770155933842e-03, /* 0xBF66C16C, 0x16BEBD93 */
    P3 =  6.61375632143793436117e-05, /* 0x3F11566A, 0xAF25DE2C */
    P4 = -1.65339022054652515390e-06, /* 0xBEBBBD41, 0xC5D26BF1 */
    P5 =  4.13813679705723846039e-08, /* 0x3E663769, 0x72BEA4D0 */
    lg2     =  6.93147180559945286227e-01, /* 0x3FE62E42, 0xFEFA39EF */
    lg2_h   =  6.93147182464599609375e-01, /* 0x3FE62E43, 0x00000000 */
    lg2_l   = -1.90465429995776804525e-09, /* 0xBE205C61, 0x0CA86C39 */
    ovt     =  8.0085662595372944372e-017, /* -(1024-log2(ovfl+.5ulp)) */
    cp      =  9.61796693925975554329e-01, /* 0x3FEEC709, 0xDC3A03FD =2/(3ln2) */
    cp_h    =  9.61796700954437255859e-01, /* 0x3FEEC709, 0xE0000000 =(float)cp */
    cp_l    = -7.02846165095275826516e-09, /* 0xBE3E2FE0, 0x145B01F5 =tail of cp_h*/
    ivln2   =  1.44269504088896338700e+00, /* 0x3FF71547, 0x652B82FE =1/ln2 */
    ivln2_h =  1.44269502162933349609e+00, /* 0x3FF71547, 0x60000000 =24b 1/ln2*/
    ivln2_l =  1.92596299112661746887e-08; /* 0x3E54AE0B, 0xF85DDF44 =1/ln2 tail*/

}

/* Get two 32 bit ints from a double.  */
__forceinline constexpr void EXTRACT_WORDS(int32_t& hi, int32_t& lo, double d) noexcept {
	uint64_t i = std::bit_cast<uint64_t>(d);
	hi = i >> 32;
	lo = uint32_t(i);
}

/* Get two 32 bit ints from a double.  */
__forceinline constexpr void EXTRACT_WORDS(int32_t &hi, uint32_t &lo, double d) noexcept {
    uint64_t i = std::bit_cast<uint64_t>(d);
	hi = i >> 32;
	lo = uint32_t(i);
}

/* Set the less significant 32 bits of a double from an int.  */
__forceinline constexpr void SET_LOW_WORD(double &d, uint32_t lo) noexcept {                                             
    uint64_t i = std::bit_cast<uint64_t>(d);
    i &= 0xffffffff00000000ull;                
    i |= uint32_t(lo);                       
    d = std::bit_cast<double>(i);
}

/* Get the more significant 32 bit int from a double.  */
__forceinline constexpr void GET_HIGH_WORD(int32_t &hi, double d) noexcept {                                             
    uint64_t i = std::bit_cast<uint64_t>(d);                               
    hi = i >> 32;                            
}

/* Set the more significant 32 bits of a double from an int.  */
__forceinline constexpr void SET_HIGH_WORD(double &d, int32_t hi) noexcept {                                             
    uint64_t i = std::bit_cast<uint64_t>(d);                                
    i &= 0xffffffff;                           
    i |= uint64_t(hi) << 32;                 
    d = std::bit_cast<double>(i);                                   
}


//=--------------------------------------------------------------------------------------------------------------------
constexpr double pow_impl(double x, double y) noexcept
{
	using namespace double_pow_data;

	double z,ax,z_h,z_l,p_h,p_l;
	double y1,t1,t2,r,s,t,u,v,w;
	int32_t i,j,k,yisint,n;
	int32_t hx,hy,ix,iy;
	uint32_t lx,ly;

	EXTRACT_WORDS(hx, lx, x);
	EXTRACT_WORDS(hy, ly, y);
	ix = hx & 0x7fffffff;
	iy = hy & 0x7fffffff;

	/* x**0 = 1, even if x is NaN */
	if ((iy|ly) == 0)
		return 1.0;
	/* 1**y = 1, even if y is NaN */
	if (hx == 0x3ff00000 && lx == 0)
		return 1.0;
	/* NaN if either arg is NaN */
	if (ix > 0x7ff00000 || (ix == 0x7ff00000 && lx != 0) ||
	    iy > 0x7ff00000 || (iy == 0x7ff00000 && ly != 0))
		return x + y;

	/* determine if y is an odd int when x < 0
	 * yisint = 0       ... y is not an integer
	 * yisint = 1       ... y is an odd int
	 * yisint = 2       ... y is an even int
	 */
	yisint = 0;
	if (hx < 0) {
		if (iy >= 0x43400000)
			yisint = 2; /* even integer y */
		else if (iy >= 0x3ff00000) {
			k = (iy>>20) - 0x3ff;  /* exponent */
			if (k > 20) {
				j = ly>>(52-k);
				if (std::bit_cast<uint32_t>(j<<(52-k)) == ly) //TOOD:  warning: comparison of integer expressions of different signedness
					yisint = 2 - (j&1);
			} else if (ly == 0) {
				j = iy>>(20-k);
				if ((j<<(20-k)) == iy)
					yisint = 2 - (j&1);
			}
		}
	}

	/* special value of y */
	if (ly == 0) {
		if (iy == 0x7ff00000) {  /* y is +-inf */
			if (((ix-0x3ff00000)|lx) == 0)  /* (-1)**+-inf is 1 */
				return 1.0;
			else if (ix >= 0x3ff00000) /* (|x|>1)**+-inf = inf,0 */
				return hy >= 0 ? y : 0.0;
			else if ((ix|lx) != 0)     /* (|x|<1)**+-inf = 0,inf if x!=0 */
				return hy >= 0 ? 0.0 : -y;
		}
		if (iy == 0x3ff00000) {    /* y is +-1 */
			if (hy >= 0)
				return x;
			y = 1/x;
			return y;
		}
		if (hy == 0x40000000)    /* y is 2 */
			return x*x;
		if (hy == 0x3fe00000) {  /* y is 0.5 */
			if (hx >= 0)     /* x >= +0 */
				return cx::sqrt(x);
		}
	}

	ax = cx::fabs(x);
	/* special value of x */
	if (lx == 0) {
		if (ix == 0x7ff00000 || ix == 0 || ix == 0x3ff00000) { /* x is +-0,+-inf,+-1 */
			z = ax;
			if (hy < 0)   /* z = (1/|x|) */
				z = 1.0/z;
			if (hx < 0) {
				if (((ix-0x3ff00000)|yisint) == 0) {
					z = (z-z)/(z-z); /* (-1)**non-int is NaN */ //! warning C4723: potential divide by 0
				} else if (yisint == 1)
					z = -z;          /* (x<0)**odd = -(|x|**odd) */
			}
			return z;
		}
	}

	s = 1.0; /* sign of result */
	if (hx < 0) {
		if (yisint == 0) /* (x<0)**(non-int) is NaN */
			return (x-x)/(x-x); //! warning C4723: potential divide by 0
		if (yisint == 1) /* (x<0)**(odd int) */
			s = -1.0;
	}

	/* |y| is huge */
	if (iy > 0x41e00000) { /* if |y| > 2**31 */
		if (iy > 0x43f00000) {  /* if |y| > 2**64, must o/uflow */
			if (ix <= 0x3fefffff)
				return hy < 0 ? huge*huge : tiny*tiny;
			if (ix >= 0x3ff00000)
				return hy > 0 ? huge*huge : tiny*tiny;
		}
		/* over/underflow if x is not close to one */
		if (ix < 0x3fefffff)
			return hy < 0 ? s*huge*huge : s*tiny*tiny;
		if (ix > 0x3ff00000)
			return hy > 0 ? s*huge*huge : s*tiny*tiny;
		/* now |1-x| is tiny <= 2**-20, suffice to compute
		   log(x) by x-x^2/2+x^3/3-x^4/4 */
		t = ax - 1.0;       /* t has 20 trailing zeros */
		w = (t*t)*(0.5 - t*(0.3333333333333333333333-t*0.25));
		u = ivln2_h*t;      /* ivln2_h has 21 sig. bits */
		v = t*ivln2_l - w*ivln2;
		t1 = u + v;
		SET_LOW_WORD(t1, 0);
		t2 = v - (t1-u);
	} else {
		double ss,s2,s_h,s_l,t_h,t_l;
		n = 0;
		/* take care subnormal number */
		if (ix < 0x00100000) {
			ax *= two53;
			n -= 53;
			GET_HIGH_WORD(ix,ax);
		}
		n += ((ix)>>20) - 0x3ff;
		j = ix & 0x000fffff;
		/* determine interval */
		ix = j | 0x3ff00000;   /* normalize ix */
		if (j <= 0x3988E)      /* |x|<sqrt(3/2) */
			k = 0;
		else if (j < 0xBB67A)  /* |x|<sqrt(3)   */
			k = 1;
		else {
			k = 0;
			n += 1;
			ix -= 0x00100000;
		}
		SET_HIGH_WORD(ax, ix);

		/* compute ss = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5) */
		u = ax - bp[k];        /* bp[0]=1.0, bp[1]=1.5 */
		v = 1.0/(ax+bp[k]);
		ss = u*v;
		s_h = ss;
		SET_LOW_WORD(s_h, 0);
		/* t_h=ax+bp[k] High */
		t_h = 0.0;
		SET_HIGH_WORD(t_h, ((ix>>1)|0x20000000) + 0x00080000 + (k<<18));
		t_l = ax - (t_h-bp[k]);
		s_l = v*((u-s_h*t_h)-s_h*t_l);
		/* compute log(ax) */
		s2 = ss*ss;
		r = s2*s2*(L1+s2*(L2+s2*(L3+s2*(L4+s2*(L5+s2*L6)))));
		r += s_l*(s_h+ss);
		s2 = s_h*s_h;
		t_h = 3.0 + s2 + r;
		SET_LOW_WORD(t_h, 0);
		t_l = r - ((t_h-3.0)-s2);
		/* u+v = ss*(1+...) */
		u = s_h*t_h;
		v = s_l*t_h + t_l*ss;
		/* 2/(3log2)*(ss+...) */
		p_h = u + v;
		SET_LOW_WORD(p_h, 0);
		p_l = v - (p_h-u);
		z_h = cp_h*p_h;        /* cp_h+cp_l = 2/(3*log2) */
		z_l = cp_l*p_h+p_l*cp + dp_l[k];
		/* log2(ax) = (ss+..)*2/(3*log2) = n + dp_h + z_h + z_l */
		t = (double)n;
		t1 = ((z_h + z_l) + dp_h[k]) + t;
		SET_LOW_WORD(t1, 0);
		t2 = z_l - (((t1 - t) - dp_h[k]) - z_h);
	}

	/* split up y into y1+y2 and compute (y1+y2)*(t1+t2) */
	y1 = y;
	SET_LOW_WORD(y1, 0);
	p_l = (y-y1)*t1 + y*t2;
	p_h = y1*t1;
	z = p_l + p_h;
	EXTRACT_WORDS(j, i, z);
	if (j >= 0x40900000) {                      /* z >= 1024 */
		if (((j-0x40900000)|i) != 0)        /* if z > 1024 */
			return s*huge*huge;         /* overflow */
		if (p_l + ovt > z - p_h)
			return s*huge*huge;         /* overflow */
	} else if ((j&0x7fffffff) >= 0x4090cc00) {  /* z <= -1075 */  // FIXME: instead of abs(j) use unsigned j
		if (((j-0xc090cc00)|i) != 0)        /* z < -1075 */
			return s*tiny*tiny;         /* underflow */
		if (p_l <= z - p_h)
			return s*tiny*tiny;         /* underflow */
	}
	/*
	 * compute 2**(p_h+p_l)
	 */
	i = j & 0x7fffffff;
	k = (i>>20) - 0x3ff;
	n = 0;
	if (i > 0x3fe00000) {  /* if |z| > 0.5, set n = [z+0.5] */
		n = j + (0x00100000>>(k+1));
		k = ((n&0x7fffffff)>>20) - 0x3ff;  /* new k for n */
		t = 0.0;
		SET_HIGH_WORD(t, n & ~(0x000fffff>>k));
		n = ((n&0x000fffff)|0x00100000)>>(20-k);
		if (j < 0)
			n = -n;
		p_h -= t;
	}
	t = p_l + p_h;
	SET_LOW_WORD(t, 0);
	u = t*lg2_h;
	v = (p_l-(t-p_h))*lg2 + t*lg2_l;
	z = u + v;
	w = v - (z-u);
	t = z*z;
	t1 = z - t*(P1+t*(P2+t*(P3+t*(P4+t*P5))));
	r = (z*t1)/(t1-2.0) - (w + z*w);
	z = 1.0 - (r-z);
	GET_HIGH_WORD(j, z);
	j += n<<20;
	if ((j>>20) <= 0)  /* subnormal output */
		z = cx::scalbn(z,n);
	else
		SET_HIGH_WORD(z, j);
	return s*z;
}


} // namespace detail

/// xʸ, x^y, x to the power of y
constexpr float powf(float x, float y) noexcept { return detail::powf_impl(x, y); }
/// xʸ, x^y, x to the power of y
constexpr float pow(float x, float y) noexcept { return detail::powf_impl(x, y); }
/// xʸ, x^y, x to the power of y
constexpr double pow(double x, double y) noexcept { return detail::pow_impl(x,y); }


/// xʸ, x^y, x to the power of y
constexpr float vm_powf(float x, float y) noexcept { return detail::pow_template_f(x, y); } //TODO: name !
/// xʸ, x^y, x to the power of y
constexpr float vm_pow(float x, float y) noexcept { return detail::pow_template_f(x, y); } //TODO: name !
/// xʸ, x^y, x to the power of y
constexpr double vm_pow(double x, double y) noexcept { return detail::pow_template_d(x, y); } //TODO: name !

namespace detail { //TODO: // vm::log


static constexpr double log_special_cases(double x1, double r) {
    double res = r;
    bool overflow = !is_finite(x1);
    bool underflow = x1 < VM_SMALLEST_NORMAL;  // denormals not supported by this functions
    if (!horizontal_or(overflow | underflow)) {
        return res;                              // normal path
    }
    // overflow and underflow
    res = select(underflow, nan_vec<double>(NAN_LOG), res);                // x1  < 0 gives NAN
    res = select(is_zero_or_subnormal(x1), -infinite_vec<double>(), res);  // x1 == 0 gives -INF
    res = select(overflow, x1, res);                                      // INF or NAN goes through
    res = select(is_inf(x1) & sign_bit(x1), nan_vec<double>(NAN_LOG), res);// -INF gives NAN
    return res;
}

static constexpr float log_special_cases(float x1, float r) {
    float res = r;
    bool overflow = !is_finite(x1);
    bool underflow = x1 < VM_SMALLEST_NORMALF; // denormals not supported by this functions
    if (!horizontal_or(overflow | underflow)) {
        return res;                              // normal path
    }
    // overflow and underflow
    res = select(underflow, nan_vec<float>(NAN_LOG), res);                // x1  < 0 gives NAN
    res = select(is_zero_or_subnormal(x1), -infinite_vec<float>(), res);  // x1 == 0 gives -INF
    res = select(overflow, x1, res);                                      // INF or NAN goes through
    res = select(is_inf(x1) & sign_bit(x1), nan_vec<float>(NAN_LOG), res);// -INF gives NAN
    return res;
}

// function to_float: convert integer vector to float vector
static constexpr float to_float(int32_t a) {
    return float(a); // _mm_cvtepi32_ps(a);
}

// log function, double precision
// template parameters:
// VTYPE:  f.p. vector type
// M1: 0 for log, 1 for log1p
template<typename VTYPE, int M1>
static constexpr VTYPE log_d(VTYPE const initial_x) {
    // define constants
    const double ln2_hi =  0.693359375;
    const double ln2_lo = -2.121944400546905827679E-4;
    const double P0log  =  7.70838733755885391666E0;
    const double P1log  =  1.79368678507819816313E1;
    const double P2log  =  1.44989225341610930846E1;
    const double P3log  =  4.70579119878881725854E0;
    const double P4log  =  4.97494994976747001425E-1;
    const double P5log  =  1.01875663804580931796E-4;
    const double Q0log  =  2.31251620126765340583E1;
    const double Q1log  =  7.11544750618563894466E1;
    const double Q2log  =  8.29875266912776603211E1;
    const double Q3log  =  4.52279145837532221105E1;
    const double Q4log  =  1.12873587189167450590E1;

    VTYPE  x1, x, x2, px, qx, res, fe;           // data vectors

    if constexpr (M1 == 0) {
        x1 = initial_x;                          // log(x)
    }
    else {
        x1 = initial_x + 1.0;                    // log(x+1)
    }
    // separate mantissa from exponent
    // VTYPE x  = fraction(x1) * 0.5;
    x  = fraction_2(x1);
    fe = exponent_f(x1);

    auto blend = x > VM_SQRT2*0.5;               // boolean vector
    x  = if_add(!blend, x, x);                   // conditional add
    fe = if_add(blend, fe, 1.);                  // conditional add

    if constexpr (M1 == 0) {
        // log(x). Expand around 1.0
        x -= 1.0;
    }
    else {
        // log(x+1). Avoid loss of precision when adding 1 and later subtracting 1 if exponent = 0
        x = select(fe==0., initial_x, x - 1.0);
    }

    // rational form
    px  = polynomial_5 (x, P0log, P1log, P2log, P3log, P4log, P5log);
    x2  = x * x;
    px *= x * x2;
    qx  = polynomial_5n(x, Q0log, Q1log, Q2log, Q3log, Q4log);
    res = px / qx ;

    // add exponent
    res  = mul_add(fe, ln2_lo, res);             // res += fe * ln2_lo;
    res += nmul_add(x2, 0.5, x);                 // res += x  - 0.5 * x2;
    res  = mul_add(fe, ln2_hi, res);             // res += fe * ln2_hi;
#ifdef SIGNED_ZERO                               // pedantic preservation of signed zero
    res = select(initial_x == 0., initial_x, res);
#endif
    // handle special cases, or return res
    return log_special_cases(x1, res);
}

// log function, single precision
// template parameters:
// VTYPE:  f.p. vector type
// M1: 0 for log, 1 for log1p
template<typename VTYPE, int M1>
static constexpr VTYPE log_f(VTYPE const initial_x) {

	// define constants
	const float ln2f_hi = 0.693359375f;
	const float ln2f_lo = -2.12194440E-4f;
	const float P0logf = 3.3333331174E-1f;
	const float P1logf = -2.4999993993E-1f;
	const float P2logf = 2.0000714765E-1f;
	const float P3logf = -1.6668057665E-1f;
	const float P4logf = 1.4249322787E-1f;
	const float P5logf = -1.2420140846E-1f;
	const float P6logf = 1.1676998740E-1f;
	const float P7logf = -1.1514610310E-1f;
	const float P8logf = 7.0376836292E-2f;

	VTYPE  x1, x, res, x2, fe;                   // data vectors

	if constexpr (M1 == 0) {
		x1 = initial_x;                          // log(x)
	}
	else {
		x1 = initial_x + 1.0f;                   // log(x+1)
	}

	// separate mantissa from exponent
	x = fraction_2(x1);
	auto e = exponent(x1);                       // integer vector

	auto blend = x > float(VM_SQRT2 * 0.5);        // boolean vector
	x = if_add(!blend, x, x);                   // conditional add
	e = if_add(decltype(e > e)(blend), e, decltype(e)(1));  // conditional add
	fe = to_float(e);

	if constexpr (M1 == 0) {
		// log(x). Expand around 1.0
		x -= 1.0f;
	}
	else {
		// log(x+1). Avoid loss of precision when adding 1 and later subtracting 1 if exponent = 0
		x = select(decltype(x > x)(e == 0), initial_x, x - 1.0f);
	}

	// Taylor expansion
	res = polynomial_8(x, P0logf, P1logf, P2logf, P3logf, P4logf, P5logf, P6logf, P7logf, P8logf);
	x2 = x * x;
	res *= x2 * x;

	// add exponent
	res = mul_add(fe, ln2f_lo, res);            // res += ln2f_lo  * fe;
	res += nmul_add(x2, 0.5f, x);                // res += x - 0.5f * x2;
	res = mul_add(fe, ln2f_hi, res);            // res += ln2f_hi  * fe;
#ifdef SIGNED_ZERO                               // pedantic preservation of signed zero
	res = select(initial_x == 0.f, initial_x, res);
#endif
	// handle special cases, or return res
	return log_special_cases(x1, res);
}


} // namespace detail 

namespace detail {

// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// 
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
constexpr double log_impl(double x) noexcept 
{
    constexpr double ln2_hi = 6.93147180369123816490e-01;
    constexpr double ln2_lo = 1.90821492927058770002e-10;
    constexpr double two54 = 1.80143985094819840000e+16;
    constexpr double Lg1 = 6.666666666666735130e-01;
    constexpr double Lg2 = 3.999999999940941908e-01;
    constexpr double Lg3 = 2.857142874366239149e-01;
    constexpr double Lg4 = 2.222219843214978396e-01;
    constexpr double Lg5 = 1.818357216161805012e-01;
    constexpr double Lg6 = 1.531383769920937332e-01;
    constexpr double Lg7 = 1.479819860511658591e-01; 
    constexpr double zero = 0.0;

    double hfsq, f, s, z, R, w, t1, t2, dk;
    int32_t k, i, j;

    int64_t ax = std::bit_cast<int64_t> ( x );
    int32_t hx = static_cast<int32_t>( ax >> 32 );
    uint32_t lx = static_cast<uint32_t>( ax );

    k = 0;
    if ( hx < 0x00100000 )
    {
        // x < 2**-1022  
        if ( ( ( hx & 0x7fffffff ) | lx ) == 0 )
        {
            // log(+-0)=-inf 
            //return -two54 / zero;
            return -std::numeric_limits<double>::infinity( );
        }
        if ( hx < 0 )
        {
            // log(-#) = NaN 
            //return ( x - x ) / zero;
            return std::numeric_limits<double>::quiet_NaN( );
        }
        // subnormal number, scale up x 
        k -= 54; 
        x *= two54; 
        ax = std::bit_cast<int64_t> ( x );
        hx = static_cast<int32_t>( ax >> 32 );
    }
    if ( hx >= 0x7ff00000 )
    {
        return x + x;
    }
    k += ( hx >> 20 ) - 1023;
    hx &= 0x000fffff;
    i = ( hx + 0x95f64 ) & 0x100000;
    // normalize x or x/2 
    ax = (static_cast<int64_t>( hx | ( i ^ 0x3ff00000 ) ) << 32) | (ax & 0x00000000FFFFFFFF);
    x = std::bit_cast<double>( ax );
    
    k += ( i >> 20 );
    f = x - 1.0;
    if ( ( 0x000fffff & ( 2 + hx ) ) < 3 )
    {	
        // -2**-20 <= f < 2**-20 
        if ( f == zero )
        {
            if ( k == 0 )
            {
                return zero;
            }
            else
            {
                dk = static_cast<double>(k);
                return dk * ln2_hi + dk * ln2_lo;
            }
        }
        R = f * f * ( 0.5 - 0.33333333333333333 * f );
        if ( k == 0 )
        {
            return f - R;
        }
        else
        {
            dk = static_cast<double>( k );
            return dk * ln2_hi - ( ( R - dk * ln2_lo ) - f );
        }
    }
    s = f / ( 2.0 + f );
    dk = static_cast<double>( k );
    z = s * s;
    i = hx - 0x6147a;
    w = z * z;
    j = 0x6b851 - hx;
    t1 = w * ( Lg2 + w * ( Lg4 + w * Lg6 ) );
    t2 = z * ( Lg1 + w * ( Lg3 + w * ( Lg5 + w * Lg7 ) ) );
    i |= j;
    R = t2 + t1;
    if ( i > 0 )
    {
        hfsq = 0.5 * f * f;
        if ( k == 0 ) 
            return f - ( hfsq - s * ( hfsq + R ) ); 
        else
            return dk * ln2_hi - ( ( hfsq - ( s * ( hfsq + R ) + dk * ln2_lo ) ) - f );
    }
    else
    {
        if ( k == 0 ) 
            return f - s * ( f - R ); 
        else
            return dk * ln2_hi - ( ( s * ( f - R ) - dk * ln2_lo ) - f );
    }
}


// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// 
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
constexpr float log_impl(float x) noexcept 
{
    constexpr float ln2_hi = 6.9313812256e-01f;
    constexpr float ln2_lo = 9.0580006145e-06f;
    constexpr float two25 = 3.355443200e+07f;
    /* |(log(1+s)-log(1-s))/s - Lg(s)| < 2**-34.24 (~[-4.95e-11, 4.97e-11]). */
    constexpr float Lg1 = 0xaaaaaa.0p-24f;
    constexpr float Lg2 = 0xccce13.0p-25f;
    constexpr float Lg3 = 0x91e9ee.0p-25f;
    constexpr float Lg4 = 0xf89e26.0p-26f;

    constexpr float zero = 0.0f;

    float hfsq, f, s, z, R, w, t1, t2, dk;
    //int32_t k, ix, i, j;

    int32_t ix = std::bit_cast<int32_t>( x );

    int32_t k = 0;
    if ( ix < 0x00800000 )
    {
        // x < 2**-126  
        if ( ( ix & 0x7fffffff ) == 0 )
        {
            // log(+-0)=-inf 
            //return -two25 / zero;
            return -std::numeric_limits<float>::infinity( );
        }
        if ( ix < 0 )
        {
            // log(-#) = NaN 
            //return ( x - x ) / zero;
            return std::numeric_limits<float>::quiet_NaN( );
        }
        // subnormal number, scale up x 
        k -= 25; 
        x *= two25; 
        ix = std::bit_cast<int32_t>( x );
    }
    if ( ix >= 0x7f800000 )
    {
        return x + x;
    }
    k += ( ix >> 23 ) - 127;
    ix &= 0x007fffff;
    int32_t i = ( ix + ( 0x95f64 << 3 ) ) & 0x800000;

    // normalize x or x/2 
    x = std::bit_cast<float>( ix | ( i ^ 0x3f800000 ) );
    
    k += ( i >> 23 );
    f = x - 1.0f;
    if ( ( 0x007fffff & ( 0x8000 + ix ) ) < 0xc000 )
    {	
        // -2**-9 <= f < 2**-9 
        if ( f == zero )
        {
            if ( k == 0 )
            {
                return zero;
            }
            else
            {
                dk = static_cast<float>( k );
                return dk * ln2_hi + dk * ln2_lo;
            }
        }
        R = f * f * ( 0.5f - (float)0.33333333333333333 * f );
        if ( k == 0 ) return f - R; else
        {
            dk = static_cast<float>( k );
            return dk * ln2_hi - ( ( R - dk * ln2_lo ) - f );
        }
    }
    s = f / ( 2.0f + f );
    dk = static_cast<float>( k );
    z = s * s;
    i = ix - ( 0x6147a << 3 );
    w = z * z;
    int32_t j = ( 0x6b851 << 3 ) - ix;
    t1 = w * ( Lg2 + w * Lg4 );
    t2 = z * ( Lg1 + w * Lg3 );
    i |= j;
    R = t2 + t1;
    if ( i > 0 )
    {
        hfsq = 0.5f * f * f;
        if ( k == 0 ) 
            return f - ( hfsq - s * ( hfsq + R ) ); 
        else
            return dk * ln2_hi - ( ( hfsq - ( s * ( hfsq + R ) + dk * ln2_lo ) ) - f );
    }
    else
    {
        if ( k == 0 ) 
            return f - s * ( f - R ); 
        else
            return dk * ln2_hi - ( ( s * ( f - R ) - dk * ln2_lo ) - f );
    }
}

} // namespace detail

/// logₑ(x)  Computes the natural (base e) logarithm of x
constexpr float  logf(float x) noexcept { return detail::log_impl(x); }

/// logₑ(x)  Computes the natural (base e) logarithm of x
constexpr float  log(float x) noexcept { return detail::log_impl(x); }

/// logₑ(x)  Computes the natural (base e) logarithm of x
constexpr double log(double x) noexcept { return detail::log_impl(x); }

/// logₑ(x)
constexpr float vm_log(float x) noexcept { return detail::log_f<float, 0>(x); }
/// logₑ(x)
constexpr double vm_log(double x) noexcept { return detail::log_d<double, 0>(x); }

/// logₑ(1+x)    more precise than the expression log(1+x) if x is close to zero.
constexpr float log1p(float x) noexcept { return detail::log_f<float, 3>(x); }
/// logₑ(1+x)    more precise than the expression log(1+x) if x is close to zero.
constexpr double log1p(double x) noexcept { return detail::log_d<double, 3>(x); }

/// logᵦ(x)
constexpr float log(float base, float x) noexcept { return detail::log_f<float, 0>(x) / detail::log_f<float, 0>(base); }  //TODO: low precision, slow speed?
/// logᵦ(x)
constexpr double log(double base, double x) noexcept { return detail::log_d<double, 0>(x) / detail::log_d<double, 0>(base); }  //TODO: low precision, slow speed?

/// log₂(x)
constexpr float log2(float x) noexcept { return LOG2E<float> * detail::log_f<float, 0>(x); } //TODO: low precision?
/// log₂(x)
constexpr double log2(double x) noexcept { return LOG2E<float> * detail::log_d<double, 0>(x); } //TODO: low precision?

/// log₃(x)
constexpr float log3(float x) noexcept { return LOG3E<float> * detail::log_f<float, 0>(x); } //TODO: low precision?
/// log₃(x)
constexpr double log3(double x) noexcept { return LOG3E<double> * detail::log_d<double, 0>(x); } //TODO: low precision?

/// log₄(x)
constexpr float log4(float x) noexcept { return LOG4E<float> *detail::log_f<float, 0>(x); } //TODO: low precision?
/// log₄(x)
constexpr double log4(double x) noexcept { return LOG4E<double> *detail::log_d<double, 0>(x); } //TODO: low precision?

/// log₅(x)
constexpr float log5(float x) noexcept { return LOG5E<float> *detail::log_f<float, 0>(x); } //TODO: low precision?
/// log₅(x)
constexpr double log5(double x) noexcept { return LOG5E<double> *detail::log_d<double, 0>(x); } //TODO: low precision?

/// log₆(x)
constexpr float log6(float x) noexcept { return LOG6E<float> *detail::log_f<float, 0>(x); } //TODO: low precision?
/// log₆(x)
constexpr double log6(double x) noexcept { return LOG6E<double> *detail::log_d<double, 0>(x); } //TODO: low precision?

/// log₇(x)
constexpr float log7(float x) noexcept { return LOG6E<float> *detail::log_f<float, 0>(x); } //TODO: low precision?
/// log₇(x)
constexpr double log7(double x) noexcept { return LOG6E<double> *detail::log_d<double, 0>(x); } //TODO: low precision?

/// log₈(x)
constexpr float log8(float x) noexcept { return LOG8E<float> *detail::log_f<float, 0>(x); } //TODO: low precision?
/// log₈(x)
constexpr double log8(double x) noexcept { return LOG8E<double> *detail::log_d<double, 0>(x); } //TODO: low precision?

/// log₉(x)
constexpr float log9(float x) noexcept { return LOG9E<float> *detail::log_f<float, 0>(x); } //TODO: low precision?
/// log₉(x)
constexpr double log9(double x) noexcept { return LOG9E<double> *detail::log_d<double, 0>(x); } //TODO: low precision?

/// log₁₀(x)
constexpr float log10(float x) noexcept { return LOG10E<float> * detail::log_f<float, 0>(x); } //TODO: low precision?
/// log₁₀(x)
constexpr double log10(double x) noexcept { return LOG10E<double> * detail::log_d<double, 0>(x); } //TODO: low precision?

namespace detail { //TODO: // vm::exp

// This function calculates pow(2,n) where n must be an integer. Does not check for overflow or underflow
static constexpr double vm_pow2n(double n) noexcept {
    const double pow2_52 = 4503599627370496.0;   // 2^52
    const double bias = 1023.0;                  // bias in exponent
    double a = n + (bias + pow2_52);              // put n + bias in least significant bits
    int64_t b = reinterpret_i(a);                  // bit-cast to integer
    int64_t c = b << 52;                           // shift left 52 places to get into exponent field
    double d = reinterpret_d(c);                  // bit-cast back to double
    return d;
}

static constexpr float vm_pow2n (float n) noexcept {
    const float pow2_23 =  8388608.0;            // 2^23
    const float bias = 127.0;                    // bias in exponent
    float a = n + (bias + pow2_23);              // put n + bias in least significant bits
    int32_t b = reinterpret_i(a);                  // bit-cast to integer
    int32_t c = b << 23;                           // shift left 23 places to get into exponent field
    float d = reinterpret_f(c);                  // bit-cast back to float
    return d;
}

// Template for exp function, double precision
// The limit of abs(x) is defined by max_x below
// This function does not produce denormals
// Template parameters:
// VTYPE:  double vector type
// M1: 0 for exp, 1 for expm1
// BA: 0 for exp, 1 for 0.5*exp, 2 for pow(2,x), 10 for pow(10,x)
// Taylor expansion
template<typename VTYPE, int M1, int BA>
static constexpr VTYPE exp_d(VTYPE const initial_x) noexcept {

    // Taylor coefficients, 1/n!
    // Not using minimax approximation because we prioritize precision close to x = 0
    const double p2  = 1./2.;
    const double p3  = 1./6.;
    const double p4  = 1./24.;
    const double p5  = 1./120.;
    const double p6  = 1./720.;
    const double p7  = 1./5040.;
    const double p8  = 1./40320.;
    const double p9  = 1./362880.;
    const double p10 = 1./3628800.;
    const double p11 = 1./39916800.;
    const double p12 = 1./479001600.;
    const double p13 = 1./6227020800.;

    // maximum abs(x), value depends on BA, defined below
    // The lower limit of x is slightly more restrictive than the upper limit.
    // We are specifying the lower limit, except for BA = 1 because it is not used for negative x
    double max_x;

    // data vectors
    VTYPE  x, r, z, n2;

    if constexpr (BA <= 1) { // exp(x)
        max_x = BA == 0 ? 708.39 : 709.7;        // lower limit for 0.5*exp(x) is -707.6, but we are using 0.5*exp(x) only for positive x in hyperbolic functions
        const double ln2d_hi = 0.693145751953125;
        const double ln2d_lo = 1.42860682030941723212E-6;
        x  = initial_x;
        r  = round(initial_x*VM_LOG2E);
        // subtraction in two steps for higher precision
        x = nmul_add(r, ln2d_hi, x);             //  x -= r * ln2d_hi;
        x = nmul_add(r, ln2d_lo, x);             //  x -= r * ln2d_lo;
    }
    else if constexpr (BA == 2) { // pow(2,x)
        max_x = 1022.0;
        r  = round(initial_x);
        x  = initial_x - r;
        x *= VM_LN2;
    }
    else if constexpr (BA == 10) { // pow(10,x)
        max_x = 307.65;
        const double log10_2_hi = 0.30102999554947019; // log10(2) in two parts
        const double log10_2_lo = 1.1451100899212592E-10;
        x  = initial_x;
        r  = round(initial_x*(VM_LOG2E*VM_LN10));
        // subtraction in two steps for higher precision
        x  = nmul_add(r, log10_2_hi, x);         //  x -= r * log10_2_hi;
        x  = nmul_add(r, log10_2_lo, x);         //  x -= r * log10_2_lo;
        x *= VM_LN10;
    }
    else  {  // undefined value of BA
        return 0.;
    }

    z = polynomial_13m(x, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);

    if constexpr (BA == 1) r--;  // 0.5 * exp(x)

    // multiply by power of 2
    n2 = vm_pow2n(r);

    if constexpr (M1 == 0) {
        // exp
        z = (z + 1.0) * n2;
    }
    else {
        // expm1
        z = mul_add(z, n2, n2 - 1.0);            // z = z * n2 + (n2 - 1.0);
#ifdef SIGNED_ZERO                               // pedantic preservation of signed zero
        z = select(initial_x == 0., initial_x, z);
#endif
    }

    // check for overflow
    auto inrange  = abs(initial_x) < max_x;      // boolean vector
    // check for INF and NAN
    inrange &= is_finite(initial_x);

    if (horizontal_and(inrange)) {
        // fast normal path
        return z;
    }
    else {
        // overflow, underflow and NAN
        r = select(sign_bit(initial_x), 0.-(M1&1), infinite_vec<VTYPE>()); // value in case of +/- overflow or INF
        z = select(inrange, z, r);                         // +/- underflow
        z = select(is_nan(initial_x), initial_x, z);       // NAN goes through
        return z;
    }
}


// Template for exp function, single precision
// The limit of abs(x) is defined by max_x below
// This function does not produce denormals
// Template parameters:
// VTYPE:  float vector type
// M1: 0 for exp, 1 for expm1
// BA: 0 for exp, 1 for 0.5*exp, 2 for pow(2,x), 10 for pow(10,x)
template<typename VTYPE, int M1, int BA>
static constexpr VTYPE exp_f(VTYPE const initial_x) noexcept {

	// Taylor coefficients
	const float P0expf = 1.f / 2.f;
	const float P1expf = 1.f / 6.f;
	const float P2expf = 1.f / 24.f;
	const float P3expf = 1.f / 120.f;
	const float P4expf = 1.f / 720.f;
	const float P5expf = 1.f / 5040.f;

	VTYPE  x, r, x2, z, n2;                      // data vectors

	// maximum abs(x), value depends on BA, defined below
	// The lower limit of x is slightly more restrictive than the upper limit.
	// We are specifying the lower limit, except for BA = 1 because it is not used for negative x
	float max_x;

	if constexpr (BA <= 1) { // exp(x)
		const float ln2f_hi = 0.693359375f;
		const float ln2f_lo = -2.12194440e-4f;
		max_x = (BA == 0) ? 87.3f : 89.0f;

		x = initial_x;
		r = round(initial_x * float(VM_LOG2E));
		x = nmul_add(r, VTYPE(ln2f_hi), x);      //  x -= r * ln2f_hi;
		x = nmul_add(r, VTYPE(ln2f_lo), x);      //  x -= r * ln2f_lo;
	}
	else if constexpr (BA == 2) {                // pow(2,x)
		max_x = 126.f;
		r = round(initial_x);
		x = initial_x - r;
		x = x * (float)VM_LN2;
	}
	else if constexpr (BA == 10) {               // pow(10,x)
		max_x = 37.9f;
		const float log10_2_hi = 0.301025391f;   // log10(2) in two parts
		const float log10_2_lo = 4.60503907E-6f;
		x = initial_x;
		r = round(initial_x * float(VM_LOG2E * VM_LN10));
		x = nmul_add(r, VTYPE(log10_2_hi), x);   //  x -= r * log10_2_hi;
		x = nmul_add(r, VTYPE(log10_2_lo), x);   //  x -= r * log10_2_lo;
		x = x * (float)VM_LN10;
	}
	else {  // undefined value of BA
		return 0.;
	}

	x2 = x * x;
	z = polynomial_5(x, P0expf, P1expf, P2expf, P3expf, P4expf, P5expf);
	z = mul_add(z, x2, x);                       // z *= x2;  z += x;

	if constexpr (BA == 1) r--;                  // 0.5 * exp(x)

	// multiply by power of 2
	n2 = vm_pow2n(r);

	if constexpr (M1 == 0) {
		// exp
		z = (z + 1.0f) * n2;
	}
	else {
		// expm1
		z = mul_add(z, n2, n2 - 1.0f);           //  z = z * n2 + (n2 - 1.0f);
#ifdef SIGNED_ZERO                               // pedantic preservation of signed zero
		z = select(initial_x == 0.f, initial_x, z);
#endif
	}

	// check for overflow
	auto inrange = abs(initial_x) < max_x;      // boolean vector
												 // check for INF and NAN
	inrange &= is_finite(initial_x);

	if (horizontal_and(inrange)) {
		// fast normal path
		return z;
	}
	else {
		// overflow, underflow and NAN
		r = select(sign_bit(initial_x), 0.f - (M1 & 1), infinite_vec<VTYPE>()); // value in case of +/- overflow or INF
		z = select(inrange, z, r);                         // +/- underflow
		z = select(is_nan(initial_x), initial_x, z);       // NAN goes through
		return z;
	}
}

} // namespace detail

namespace detail {

// ====================================================
// Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
// 
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
inline constexpr double exp_impl(double x) noexcept
{
	constexpr double one = 1.0;
	constexpr double halF[2] = {0.5,-0.5};
	constexpr double huge = 1.0e+300;
	constexpr double o_threshold = 7.09782712893383973096e+02;
	constexpr double u_threshold = -7.45133219101941108420e+02;
	constexpr double ln2HI[2] = {6.93147180369123816490e-01, -6.93147180369123816490e-01};
	constexpr double ln2LO[2] = {1.90821492927058770002e-10,  -1.90821492927058770002e-10,};
	constexpr double invln2 = 1.44269504088896338700e+00;
	constexpr double P1 = 1.66666666666666019037e-01;
	constexpr double P2 = -2.77777777770155933842e-03;
	constexpr double P3 = 6.61375632143793436117e-05;
	constexpr double P4 = -1.65339022054652515390e-06;
	constexpr double P5 = 4.13813679705723846039e-08;

	constexpr double twom1000 = 9.33263618503218878990e-302;

	double y, hi = 0.0, lo = 0.0, c, t, twopk;
	int32_t k = 0, xsb;
	int64_t ax = std::bit_cast<int64_t>(x);
	int32_t hx = static_cast<int32_t>(ax >> 32);

	// sign bit of x 
	xsb = (hx >> 31) & 1;
	// high word of |x| 
	hx &= 0x7fffffff;

	// filter out non-finite argument 
	if (hx >= 0x40862E42) {
		// if |x|>=709.78... 
		if (hx >= 0x7ff00000) {
			uint32_t lx = static_cast<uint32_t>(ax);
			if (((hx & 0xfffff) | lx) != 0)	{
				// NaN 
				return x + x;
			}
			else {
				// exp(+-inf)={inf,0} 
				return (xsb == 0) ? x : 0.0;
			}
		}
		else {
			if (x > o_threshold) {
				// overflow
				volatile double val = huge;
				return val * val;
			}
			else if (x < u_threshold) {
				// underflow 
				volatile double val = twom1000;
				return val * val;
			}
		}
	}

	// this implementation gives 2.7182818284590455 for exp(1.0),
	// which is well within the allowable error. however,
	// 2.718281828459045 is closer to the true value so we prefer that
	// answer, given that 1.0 is such an important argument value. 
	// if ( x == 1.0 ) { return 2.718281828459045235360; }

	// argument reduction 
	if (hx > 0x3fd62e42) {
		// if  |x| > 0.5 ln2 
		if (hx < 0x3FF0A2B2) {
			// and |x| < 1.5 ln2 
			hi = x - ln2HI[xsb];
			lo = ln2LO[xsb];
			k = 1 - xsb - xsb;
		}
		else {
			k = static_cast<int32_t>(invln2 * x + halF[xsb]);
			t = k;
			// t*ln2HI is exact here 
			hi = x - t * ln2HI[0];
			lo = t * ln2LO[0];
		}
		x = hi - lo;
	}
	else if (hx < 0x3e300000) {
		// when |x|<2**-28 
		if (huge + x > one) {
			// trigger inexact 
			return one + x;
		}
	}
	else k = 0;

	// x is now in primary range 
	t = x * x;
	if (k >= -1021) {
		// twopk = From32BitsTo64Bits<double>(0x3ff00000 + (k << 20), 0);
        twopk = std::bit_cast<double>(static_cast<uint64_t>(0x3ff00000 + (k << 20)) << 32);
	}
	else {
		// twopk = From32BitsTo64Bits<double>(0x3ff00000 + ((k + 1000) << 20), 0);
        twopk = std::bit_cast<double>(static_cast<uint64_t>(0x3ff00000 + ((k + 1000) << 20)) << 32);
	}
	c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
	if (k == 0) {
		return one - ((x * c) / (c - 2.0) - x);
	}
	else {
		y = one - ((lo - (x * c) / (2.0 - c)) - hi);
	}
	if (k >= -1021) {
		if (k == 1024) {
			return y * 2.0 * 0x1p1023;
		}
		return y * twopk;
	}
	else {
		return y * twopk * twom1000;
	}
}


// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// 
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
inline constexpr float exp_impl( float x ) noexcept
{
    constexpr float one = 1.0f;
    constexpr float halF[2] = { 0.5f,-0.5f };
    constexpr float huge = 1.0e+30f;
    constexpr float o_threshold = 8.8721679688e+01f;
    constexpr float u_threshold = -1.0397208405e+02f;
    constexpr float ln2HI[2] = { 6.9314575195e-01f, -6.9314575195e-01f };
    constexpr float ln2LO[2] = { 1.4286067653e-06f, -1.4286067653e-06f };
    constexpr float invln2 = 1.4426950216e+00f;

    // Domain [-0.34568, 0.34568], range ~[-4.278e-9, 4.447e-9]:
    // |x*(exp(x)+1)/(exp(x)-1) - p(x)| < 2**-27.74

    constexpr float P1 = 1.6666625440e-1f;
    constexpr float P2 = -2.7667332906e-3f;

    constexpr float twom100 = 7.8886090522e-31f;

    float y, hi = 0.0, lo = 0.0, c, t, twopk;
    int32_t k = 0, xsb;
    uint32_t hx = std::bit_cast<uint32_t>( x );

    // sign bit of x 
    xsb = ( hx >> 31 ) & 1;
    // high word of |x| 
    hx &= 0x7fffffff;

    // filter out non-finite argument 
    if ( hx >= 0x42b17218 ) {
        // if |x|>=88.721... 
        if ( hx > 0x7f800000 ) {
            // NaN 
            return x + x;
        }
        if ( hx == 0x7f800000 ) {
            // exp(+-inf)={inf,0} 
            return ( xsb == 0 ) ? x : 0.0f;
        }
        if ( x > o_threshold ) {
            // overflow 
#if defined(_MSC_VER)
            #pragma warning(push)
            #pragma warning(disable:4056)
#endif
            return huge * huge; // warning C4756: overflow in constant arithmetic
#if defined(_MSC_VER)
            #pragma warning(pop)
#endif
        }
        if ( x < u_threshold ) {
            // underflow 
            return twom100 * twom100;
        }
    }

    // argument reduction 
    if ( hx > 0x3eb17218 ) {
        // if  |x| > 0.5 ln2 
        if ( hx < 0x3F851592 ) {
            // and |x| < 1.5 ln2 
            hi = x - ln2HI[xsb];
            lo = ln2LO[xsb];
            k = 1 - xsb - xsb;
        }
        else {
            k = static_cast<int32_t>( invln2 * x + halF[xsb] );
            t = static_cast<float>( k );
            // t*ln2HI is exact here 
            hi = x - t * ln2HI[0];
            lo = t * ln2LO[0];
        }
        x = hi - lo;
    }
    else if ( hx < 0x39000000 ) {
        // when |x|<2**-14 
        if ( huge + x > one ) {
            // trigger inexact 
            return one + x;
        }
    }
    else k = 0;

    // x is now in primary range 
    t = x * x;
    if ( k >= -125 ) {
        twopk = std::bit_cast<float>( 0x3f800000 + ( k << 23 ) );
    }
    else  {
        twopk = std::bit_cast<float>( 0x3f800000 + ( ( k + 100 ) << 23 ) );
    }
    c = x - t * ( P1 + t * P2 );
    if ( k == 0 ) {
        return one - ( ( x * c ) / ( c - (float)2.0 ) - x );
    }
    else {
        y = one - ( ( lo - ( x * c ) / ( (float)2.0 - c ) ) - hi );
    }
    if ( k >= -125 ) {
        if ( k == 128 ) {
            return y * 2.0F * 0x1p127F;
        }
        return y * twopk;
    }
    else {
        return y * twopk * twom100;
    }
}

} // namespace detail

/// eˣ, e^x   e (Euler's number, 2.7182818) raised to the given power x.
constexpr float  expf(float x) noexcept { return detail::exp_impl(x); }
/// eˣ, e^x   e (Euler's number, 2.7182818) raised to the given power x.
constexpr float  exp(float x) noexcept { return detail::exp_impl(x); }
/// eˣ, e^x   e (Euler's number, 2.7182818) raised to the given power x.
constexpr double exp(double x) noexcept { return detail::exp_impl(x); }

/// eˣ, e^x   e (Euler's number, 2.7182818) raised to the given power x.
constexpr float vm_exp(float x) noexcept { return detail::exp_f<float, 0, 0>(x); } //TODO: name !
/// eˣ, e^x   e (Euler's number, 2.7182818) raised to the given power x.
constexpr double vm_exp(double x) noexcept { return detail::exp_d<double, 0, 0>(x); } //TODO: name !

/// eˣ-1    For small magnitude values of x, expm1 may be more accurate than exp(x)-1.
constexpr float expm1(float x) noexcept { return detail::exp_f<float, 3, 0>(x); }
/// eˣ-1    For small magnitude values of x, expm1 may be more accurate than exp(x)-1.
constexpr double expm1(double x) noexcept { return detail::exp_d<double, 3, 0>(x); }

/// 2ˣ    base-2 exponential function of x, which is 2 raised to the power x.
constexpr float exp2(float x) noexcept { return detail::exp_f<float, 0, 2>(x); }
/// 2ˣ    base-2 exponential function of x, which is 2 raised to the power x.
constexpr double exp2(double x) noexcept { return detail::exp_d<double, 0, 2>(x); }

/// 10ˣ    base-10 exponential function of x, which is 10 raised to the power x.
constexpr float exp10(float x) noexcept { return detail::exp_f<float, 0, 10>(x); }
/// 10ˣ    base-10 exponential function of x, which is 10 raised to the power x.
constexpr double exp10(double x) noexcept { return detail::exp_d<double, 0, 10>(x); }


/// output for solve_cubic
template<typename T>
struct solve_cubic_out {
	T x[3] = {};
	int n = 0;
};

/// Solve  a*x^3 + b*x^2 + c*x + d = 0
template<typename T>
constexpr solve_cubic_out<T> solve_cubic(T a, T b, T c, T d) {
	constexpr T cos120 = T(-0.5);
	constexpr T sin120 = T(0.8660254037844386467637231707529361834714026269051903140279034897259665084544074); //  sin(deg2rad(120))
	constexpr T epsilon = std::numeric_limits<T>::epsilon();
	solve_cubic_out<T> out;
	out.n = 0;
	if (fabs(d) < epsilon) {
		// first solution is x = 0
		out.x[out.n] = T(0.0);
		out.n += 1;
		// divide all terms by x, converting to quadratic equation
		d = c;
		c = b;
		b = a;
		a = T(0.0);
	}
	if (fabs(a) < epsilon) {
		if (fabs(b) < epsilon) {
			// linear equation
			if (fabs(c) > epsilon) {
				out.x[out.n] = -d / c;
				out.n += 1;
			}
		}
		else {
			// quadratic equation
			T yy = c * c - 4 * b * d;
			if (yy >= 0) {
				T inv2b = 1 / (2 * b);
				T y = sqrt(yy);
				out.x[out.n + 0] = (-c + y) * inv2b;
				out.x[out.n + 1] = (-c - y) * inv2b;
				out.n += 2;
			}
		}
	}
	else {
		// cubic equation
		T inva = 1 / a;
		T invaa = inva * inva;
		T bb = b * b;
		T bover3a = b * T(1 / 3.0) * inva;
		T p = (3 * a * c - bb) * T(1 / 3.0) * invaa;
		T halfq = (2 * bb * b - 9 * a * b * c + 27 * a * a * d) * T(0.5 / 27) * invaa * inva;
		T yy = p * p * p / 27 + halfq * halfq;
		if (yy > epsilon) {
			// sqrt is positive: one real solution
			T y = sqrt(yy);
			T uuu = -halfq + y;
			T vvv = -halfq - y;
			T www = fabs(uuu) > fabs(vvv) ? uuu : vvv;
			T w = cbrt(www); // T w = (www < 0) ? -pow(fabs(www), T(1 / 3.0)) : pow(www, T(1 / 3.0)); //! cbrt
			out.x[out.n] = w - p / (3 * w) - bover3a;
			out.n = 1;
		}
		else if (yy < -epsilon) {
			// sqrt is negative: three real solutions
			T x = -halfq;
			T y = sqrt(-yy);
			T theta;
			T r;
			T ux;
			T uyi;
			// convert to polar form
			if (fabs(x) > epsilon) {
				theta = (x > 0) ? atan(y / x) : (atan(y / x) + PI<T>);
				r = sqrt(x * x - yy);
			}
			else {
				// vertical line
				theta = PI<T> / T(2.0); //3.14159625f / 2;
				r = y;
			}
			// calc cube root
			theta /= T(3.0);
			r = cbrt(r); // r = pow(r, T(1 / 3.0)); //! cbrt
			// convert to complex coordinate
			ux = cos(theta) * r; //? sincos
			uyi = sin(theta) * r;
			// first solution
			out.x[out.n + 0] = ux + ux - bover3a;
			// second solution, rotate +120 degrees
			out.x[out.n + 1] = 2 * (ux * cos120 - uyi * sin120) - bover3a;
			// third solution, rotate -120 degrees
			out.x[out.n + 2] = 2 * (ux * cos120 + uyi * sin120) - bover3a;
			out.n = 3;
		}
		else {
			// sqrt is zero: two real solutions
			T www = -halfq;
			T w = cbrt(www); // T w = (www < 0) ? -pow(fabs(www), T(1 / 3.0)) : pow(www, T(1 / 3.0) ); //! cbrt
			// first solution
			out.x[out.n + 0] = w + w - bover3a;
			// second solution, rotate +120 degrees
			out.x[out.n + 1] = 2 * w * cos120 - bover3a;
			out.n = 2;
		}
	}
	return out;
}

//TODO: constexpr goto are available only in C++23 !
/// gamma function,  factorial(x-1)
constexpr float gamma(float xx) {
	if (xx > 36 || xx <= 0.0) { return INFINITY; }
	//   A single precision function to compute Gamma function
	//   by a piecewise minimax polynomial approximation.
	//
	//   The maximum relative error is 3.3E-9.0
	//
	//   The function runs 30% faster than DGAMMA,
	//   the routine provided in the Numerical Recipe.
	//
	//   Author: Fukushima, T. <Toshio.Fukushima@nao.ac.jp>
	//   Date: 2019/10/16

	float x = xx;
	float f = 1.0f;
	float g, r;
	if (x > 3.5f) {
	L1:
		x = x - 1.0f;
		f = f * x;
		if (x < 3.5f) {
			goto L10;
		}
		goto L1;
	}
	else if (x < 2.5f) {
	L2:
		if (x == 0.0f) {
			goto L9;
		}
		f = f / x;
		x = x + 1.0f;
		if (x > 2.5f) {
			goto L10;
		}
		goto L2;
	}
L10:
	if (x > 3.0f) {
		float t = x - 3.25f;
		g = +2.54925696834 + t * (+2.59257139866 + t * (+1.77691939161 + t * (+0.858823624310 + t * (+0.346281344027 + t * (+0.116231812449 + t * (+0.0343704435748))))));
	}
	else if (x < 3.0f) {
		float t = x - 2.75f;
		g = +1.60835942239 + t * (+1.31708726453 + t * (+0.891167854655 + t * (+0.384768120037 + t * (+0.155957765420 + t * (+0.0470883714066 + t * (+0.0146232242317))))));
	}
	else {
		g = 2.0f;
	}
	r = g * f;
	return r;

L9:
	r = 1.0e+38f; //? Inf ?
	return r;
}

//TODO: constexpr goto are available only in C++23 !
/// gamma function,  factorial(x-1)
/// values after 23 are approximations !
constexpr double gamma(double xx) {
	if (xx > 171 || xx <= 0) { return INFINITY; }
	//   A double precision function to compute Gamma function
	//   by a piecewise minimax polynomial approximation.
	//
	//   The (theoretical) maximum relative error is 5.5E-17,
	//
	//   being smaller than the DP machine ϵ, 1.1E-16.0
	//   The function runs 20% faster than DGAMMA,
	//   the routine provided in the Numerical Recipe.
	//
	//   Author: Fukushima, T. <Toshio.Fukushima@nao.ac.jp>
	//   Date: 2019/10/16

	double x = xx;
	double f = 1.0;
	double g, r;
	if (x > 3.5) {
	L1:
		x = x - 1.0;
		f = f * x;
		if (x < 3.5) {
			goto L10;
		}
		goto L1;
	}
	else if (x < 2.5) {
	L2:
		if (x == 0.0) {
			goto L9;
		}
		f = f / x;
		x = x + 1.0;
		if (x > 2.5) {
			goto L10;
		}
		goto L2;
	}
L10:
	if (x > 3.0) {
		double t = x - 3.25;
		g = +2.5492569667185291891 + t * (+2.5925711651299805640 + t * (+1.7769198047099783654 + t * (+0.85885382288064637892 + t * (+0.34626855336056872905 + t * (+0.11526048279921955157 + t * (+0.034350198197417378854 + t * (+0.0088826032863543439287 + t * (+0.0021498495721144269788 + t * (+0.00045576674678513314905 + t * (+0.000095940184093793519324 + t * (+0.000016738398919923317512)))))))))));
	}
	else if (x < 3.0) {
		double t = x - 2.75;
		g = +1.6083594219855455740 + t * (+1.3170871791928586304 + t * (+0.89116794802647661936 + t * (+0.38477912014818524560 + t * (+0.15595585685363435567 + t * (+0.046735160424814165697 + t * (+0.014576803693379529708 + t * (+0.0032284372223208491985 + t * (+0.00091425286320138797078 + t * (+0.00013276357775573942004 + t * (+0.000048420884483658205918 + t * (+7.3776577743984342365e-7)))))))))));
	}
	else {
		g = 2.0;
	}
	r = g * f;
	return r;

L9:
	r = 1.0e+99; //? Inf ?
	return r;
}

/// values after 23 are approximations !
constexpr double gamma(int32_t xx) { return gamma(double(xx)); }

} // namespace cx


namespace cx {
template <typename Value, size_t n>
__forceinline constexpr Value evalpoly_impl(const Value& x, const Value(&coeff)[n]) {
	Value accum = coeff[n - 1];
	for (size_t i = 1; i < n; ++i) {
		accum = fma(x, accum, coeff[n - 1 - i]);
    }
	return accum;
}

template<typename Value, typename...Ts>
__forceinline constexpr Value evalpoly(Value x, Ts ... ts) {
	 Value coeffs[]{ Value(ts)... };
	 return evalpoly_impl(x, coeffs);
}
 
namespace detail {

// Halley’s method  https://file.scirp.org/pdf/AM_2013060409554653.pdf
template<typename T>
__forceinline constexpr auto halley_lambertw(T y, T x) { // 1*exp, 2*div, 4*mul
    T ey = exp(y);         // eʸ
    T yeyx = y*ey-x;       // y*eʸ - x
    if (yeyx==0) return y;
    T y1 = y+1;
    return y - (yeyx / (y1*ey - (y+2)*yeyx/(2*y1)));
}

// 4.2  Modified methods:  http://numbers.computation.free.fr/Constants/Algorithms/newton.html
template<typename T>
__forceinline constexpr auto quartic_m4(T y, T x) { // 1*exp, 4*div, 3*idiv, 18*mul
    T ey = exp(y);         // eʸ
    T f = y*ey;           // y*eʸ
    T f_x = f - x;        // actually f is y*eʸ - x 
    if (f_x==0) return y;
    T df = f + ey;        // f' = eʸ+ yeʸ
    T ddf = df + ey;      // f'' = 2eʸ+ yeʸ
    T dddf = ddf + ey;    // f''' = 3eʸ+ yeʸ
    T ddddf = dddf + ey;  // f''' = 4eʸ+ yeʸ
    f = f_x;       
    T h = -f/df; // div
    T h2 = h*h;
    T a2 = -ddf/df; // div
    T df2 = df*df;
    T ddf2 = ddf*ddf;
    T a3 = (-df*dddf + 3*ddf2)/(df2); // div
    T a4 = (-df2*ddddf + 10*df*ddf*dddf - 15*ddf2*ddf)/(df2*df); // div
    return y + h * (1 + a2*h/2 + a3*h2/6 + a4*h*h2/24); // 3*idiv
}

// NOTICE: Input arguement is NOT z but its com <= ent def !=  as zc = z+1/e
constexpr double lambert_dw0c_ep(double zc) {
    //   50-bit accuracy computation of principal branch of Lambert W function, W_0(z),
    //   by piecewise minimax rational function approximation
    //
    //   NOTICE: Input arguement is NOT z but its com <= ent def !=  as zc = z+1/e
    //
    //   Coded by T. Fukushima <Toshio.Fukushima@nao.ac.jp>
    //
    //   Reference: T. Fukushima (2020) to be submitted
    //     "Precise and fast computation of Lambert W-functions by piecewise
    //      rational function approximation with varia <= transformation"
    double x, y, dw0c;
    if (zc < 0.0) {
       //throw(DomainError("argument out of ra >=  zc= $zc."))
       return NAN;
   } else if (zc <= 2.5498939065034735716) {        // W <= 0.893, X_1
       x=sqrt(zc);
       dw0c = evalpoly(x, -0.9999999999999998890, -2.7399668668203659304, 0.026164207726990399347, 6.370916807894900917, 7.101328651785402668, 2.9800826783006852573, 0.48819596813789865043, 0.023753035787333611915, 0.00007736576009377243100 ) /
            evalpoly(x, 1.0, 5.071610848417428005, 9.986838818354528337, 9.660755192207886908, 4.7943728991336119052, 1.1629703477704522300, 0.11849462500733755233, 0.0034326525132402226488 );
   } else if (zc <= 43.613924462669367895) {    // W <= 2.754, X_2
       x=sqrt(zc);
       dw0c = evalpoly(x, -0.99997801800578916749, -0.70415751590483602272, 2.1232260832802529071, 2.3896760702935718341, 0.77765311805029175244, 0.089686698993644741433, 0.0033062485753746403559, 0.000025106760479132851033 ) /
            evalpoly(x, 1.0, 3.0356026828085410884, 3.1434530151286777057, 1.3723156566592447275, 0.25844697415744211142, 0.019551162251819044265, 0.00048775933244530123101, 2.3165116841073152717e-6 );
   } else if (zc <= 598.45353371878276946) {    // W <= 4.821, X_3
       x=sqrt(zc);
       dw0c = evalpoly(x, -0.98967420337273506393, 0.59587680606394382748, 1.4225083018151943148, 0.44882889168323809798, 0.044504943332390033511, 0.0015218794835419578554, 0.000016072263556502220023, 3.3723373020306510843e-8 ) /
            evalpoly(x, 1.0, 1.6959402394626198052, 0.80968573415500900896, 0.14002034999817021955, 0.0093571878493790164480, 0.00023251487593389773464, 1.8060170751502988645e-6, 2.5750667337015924224e-9 );
   } else if (zc <= 8049.4919850757619109) {    // W <= 7.041, X_4
       x=sqrt(zc);
       dw0c = evalpoly(x, -0.77316491997206225517, 1.1391333504296703783, 0.43116117255217074492, 0.035773078319037507449, 0.00096441640580559092740, 8.9723854598675864757e-6, 2.5623503144117723217e-8, 1.4348813778416631453e-11 ) /
        evalpoly(x, 1.0, 0.74657287456514418083, 0.12629777033419350576, 0.0069741512959563184881, 0.00014089339244355354892, 1.0257432883152943078e-6, 2.2902687190119230940e-9, 9.2794231013264501664e-13 );
   } else if (zc <= 111124.95412121781420) {    // W <= 9.380, X_5
       x=sqrt(zc);
       dw0c = evalpoly(x, 0.12007101671553688430, 0.83352640829912822896, 0.070142775916948337582, 0.0014846357985475124849, 0.000010478757366110155290, 2.5715892987071038527e-8, 1.9384214479606474749e-11, 2.8447049039139409652e-15 ) / 
              evalpoly(x, 1.0, 0.25396738845619126630, 0.012839238907330317393, 0.00020275375632510997371, 1.1482956073449141384e-6, 2.3188370605674263647e-9, 1.4271994165742563419e-12, 1.5884836942394796961e-16 );
   } else if (zc <= 1.5870429812082297112e6) {    // W <= 11.809, X_6
       x=sqrt(zc);
       dw0c = evalpoly(x, 1.7221104439937710112, 0.39919594286484275605, 0.0079885540140685028937, 0.000042889742253257920541, 7.8146828180529864981e-8, 4.9819638764354682359e-11, 9.7650889714265294606e-15, 3.7052997281721724439e-19 ) / 
            evalpoly(x, 1.0, 0.074007438118020543008, 0.0010333501506697740545, 4.4360858035727508506e-6, 6.7822912316371041570e-9, 3.6834356707639492021e-12, 6.0836159560266041168e-16, 1.8149869335981225316e-20 );
   } else if (zc <= 2.3414708401875459509e7) {    // W <= 14.308, X_7
       x=sqrt(zc);
       dw0c = evalpoly(x, 3.7529314023434544256, 0.15491342690357806525, 0.00075663140675900784505, 1.0271609235969979059e-6, 4.7853247675930066150e-10, 7.8328040770275474410e-14, 3.9433033758391036653e-18, 3.8232862205660283978e-23 ) /
            evalpoly(x, 1.0, 0.020112985338854443555, 0.000074712286154830141768, 8.4800598003693837469e-8, 3.4182424130376911762e-11, 4.8866259139690957899e-15, 2.1223373626834634178e-19, 1.6642985671260582515e-24 );
   } else if (zc <= 3.5576474308009965225e8) {    // W <= 16.865, X_8
       x=sqrt(zc);
       dw0c = evalpoly(x, 6.0196542055606555577, 0.053496672841797864762, 0.000064340849275316501519, 2.1969090100095967485e-8, 2.5927988937033061070e-12, 1.0779198161801527308e-16, 1.3780424091017898301e-21, 3.3768973150742552802e-27 ) / 
       evalpoly(x, 1.0, 0.0052809683704233371675, 5.1020501219389558082e-6, 1.5018312292270832103e-9, 1.5677706636413188379e-13, 5.7992041238911878361e-18, 6.5133170770320780259e-23, 1.3205080139213406071e-28 );
   } else if (zc <= 5.5501716296163627854e9) {    // W <= 19.468, X_9
       x=sqrt(zc);
       dw0c = evalpoly(x, 8.4280268500989701597, 0.017155758546279713315, 5.0836620669829321508e-6, 4.3354903691832581802e-10, 1.2841017145645583385e-14, 1.3419106769745885927e-19, 4.3101698455492225750e-25, 2.6422433422088187549e-31 ) / 
       evalpoly(x, 1.0, 0.0013572006754595300315, 3.3535243481426203694e-7, 2.5206969246421264128e-11, 6.7136226273060530496e-16, 6.3324226680854686574e-21, 1.8128167400013774194e-26, 9.3662030058136796889e-33 );
   } else if (zc <= 8.8674704839657775331e10) {   // W <= 22.112, X_10
       x=sqrt(zc);
       dw0c = evalpoly(x, 10.931063230472498189, 0.0052224234540245532982, 3.7996105711810129682e-7, 8.0305793533410355824e-12, 5.9139785627090605866e-17, 1.5382020359533028724e-22, 1.2288944126268109432e-28, 1.8665089270660122398e-35 ) / 
            evalpoly(x, 1.0, 0.00034328702551197577797, 2.1395351518538844476e-8, 4.0524170186631594159e-13, 2.7181424315335710420e-18, 6.4538986638355490894e-24, 4.6494613785888987942e-30, 6.0442024367299387616e-37 );
   } else if (zc <= 1.4477791865272902816e12) {   // W <= 24.791, X_11
       x=sqrt(zc);
       dw0c = evalpoly(x, 13.502943080893871412, 0.0015284636506346264572, 2.7156967358262346166e-8, 1.4110394051242161772e-13, 2.5605734311219728461e-19, 1.6421293724425337463e-25, 3.2324944691435843553e-32, 1.2054662641251783155e-39 ) / 
            evalpoly(x, 1.0, 0.000085701512879089462255, 1.3311244435752691563e-9, 6.2788924440385347269e-15, 1.0483788152252204824e-20, 6.1943499966249160886e-27, 1.1101567860340917294e-33, 3.5897381128308962590e-41 );
   } else if (zc <= 2.4111458632511851931e13) {   // W <= 27.500, X_12
       x=sqrt(zc);
       dw0c = evalpoly(x, 16.128076167439014775, 0.00043360385176467069131, 1.8696403871820916466e-9, 2.3691795766901486045e-15, 1.0503191826963154893e-21, 1.6461927573606764263e-28, 7.9138276083474522931e-36, 7.1845890343701668760e-44 ) / 
            evalpoly(x, 1.0, 0.000021154255263102938752, 8.1006115442323280538e-11, 9.4155986022169905738e-17, 3.8725127902295302254e-23, 5.6344651115570565066e-30, 2.4860951084210029191e-37, 1.9788304737427787405e-45 );
   } else if (zc <= 4.0897036442600845564e14) {   // W <= 30.236, X_13
       x=sqrt(zc);
       dw0c = evalpoly(x, 18.796301105534486604, 0.00011989443339646469157, 1.2463377528676863250e-10, 3.8219456858010368172e-17, 4.1055693930252083265e-24, 1.5595231456048464246e-31, 1.8157173553077986962e-39, 3.9807997764326166245e-48 ) / 
            evalpoly(x, 1.0, 5.1691031988359922329e-6, 4.8325571823313711932e-12, 1.3707888746916928107e-18, 1.3754560850024480337e-25, 4.8811882975661805184e-33, 5.2518641828170201894e-41, 1.0192119593134756440e-49 );
   } else if (zc <= 7.0555901476789972402e15) {   // W <= 32.996, X_14
       x=sqrt(zc);
       dw0c = evalpoly(x, 21.500582830667332906, 0.000032441943237735273768, 8.0764963416837559148e-12, 5.9488445506122883523e-19, 1.5364106187215861531e-26, 1.4033231297002386995e-34, 3.9259872712305770430e-43, 2.0629086382257737517e-52 ) / 
            evalpoly(x, 1.0, 1.2515317642433850197e-6, 2.8310314214817074806e-13, 1.9423666416123637998e-20, 4.7128616004157359714e-28, 4.0433347391839945960e-36, 1.0515141443831187271e-44, 4.9316490935436927307e-54 );
   } else if (zc <= 1.2366607557976727287e17) {   // W <= 35.779, X_15
       x=sqrt(zc);
       dw0c = evalpoly(x, 24.235812532416977267, 8.6161505995776802509e-6, 5.1033431561868273692e-13, 8.9642393665849638164e-21, 5.5254364181097420777e-29, 1.2045072724050605792e-37, 8.0372997176526840184e-47, 1.0049140812146492611e-56 ) / 
            evalpoly(x, 1.0, 3.0046761844749477987e-7, 1.6309104270855463223e-14, 2.6842271030298931329e-22, 1.5619672632458881195e-30, 3.2131689030397984274e-39, 2.0032396245307684134e-48, 2.2520274554676331938e-58 );
   } else if (zc <= 2.1999373487930999775e18) {   // W <= 38.582, X_16
       x=sqrt(zc);
       dw0c = evalpoly(x, 26.998134347987436511, 2.2512257767572285866e-6, 3.1521230759866963941e-14, 1.3114035719790631541e-22, 1.9156784033962366146e-31, 9.8967003053444799163e-41, 1.5640423898448433548e-50, 4.6216193040664872606e-61 ) / 
            evalpoly(x, 1.0, 7.1572676370907573898e-8, 9.2500506091115760826e-16, 3.6239819582787573031e-24, 5.0187712493800424118e-33, 2.4565861988218069039e-42, 3.6435658433991660284e-52, 9.7432490640155346004e-63 );
   } else if (zc <= 3.9685392198344016155e19) {   // W <= 41.404, X_17
       x=sqrt(zc);
       dw0c = evalpoly(x, 29.784546702831970770, 5.7971764392171329944e-7, 1.9069872792601950808e-15, 1.8668700870858763312e-24, 6.4200510953370940075e-34, 7.8076624650818968559e-44, 2.9029638696956315654e-54, 2.0141870458566179853e-65 ) / 
            evalpoly(x, 1.0, 1.6924463180469706372e-8, 5.1703934311254540111e-17, 4.7871532721560069095e-26, 1.5664405832545149368e-35, 1.8113137982381331398e-45, 6.3454150289495419529e-56, 4.0072964025244397967e-67 );
   } else if (zc <= 1.4127075145274652069e104) {   // W <= 234.358, U_18
       y=log(zc); //? log 
       dw0c = evalpoly(y, 0.74413499460126776143, 0.41403243618005911160, 0.26012564166773416170, 0.021450457095960295520, 0.00051872377264705907577, 4.3574693568319975996e-6, 1.2363066058921706716e-8, 9.0194147766309957537e-12) / 
            evalpoly(y, 1.0, 0.33487811067467010907, 0.023756834394570626395, 0.00054225633008907735160, 4.4378980052579623037e-6, 1.2436585497668099330e-8, 9.0225825867631852215e-12, -4.2057836270109716654e-19 );
   } else {    //   U_19
       y=log(zc); //? log 
       dw0c = evalpoly(y, -0.61514412812729761526, 0.67979310133630936580, 0.089685353704585808963, 0.0015644941483989379249, 7.7349901878176351162e-6, 1.2891647546699435229e-8, 7.0890325988973812656e-12, 9.8419790334279711453e-16) / 
            evalpoly(y, 1.0, 0.097300263710401439315, 0.0016103672748442058651, 7.8247741003077000012e-6, 1.2949261308971345209e-8, 7.0986911219342827130e-12, 9.8426285042227044979e-16, -1.5960147252606055352e-24 );
   }
   return dw0c;
}

// NOTICE:  == ired are two input arguements z and its com <= ent def !=  as zc = z+1/e
constexpr double lambert_dwm1c_ep(double z, double zc) {
    //   50-bit accuracy computation of secondary branch of Lambert W function, W_-1(z),
    //   by piecewise minimax rational function approximation
    //
    //   NOTICE:  == ired are two input arguements z and its com <= ent def !=  as zc = z+1/e
    //
    //   Coded by T. Fukushima <Toshio.Fukushima@nao.ac.jp>
    //
    //   Reference: T. Fukushima (2020) to be submitted
    //     "Precise and fast computation of Lambert W-functions by piecewise
    //      rational function approximation with varia <= transformation"
 
    double z0 = -0.36787944117144232160;    // z0 = -1/e
    double x0 = +0.60653065971263342360;    // x0 = sqrt(1/e)
    double x, u, dwm1c;
    if (zc < 0.0) {
       // throw(DomainError("argument out of ra >=  zc= $zc."))
       return NAN;
   } else if (z <= -0.3542913309442164) {          // W >= -1.3, X_-1
       x=sqrt(zc);
       dwm1c = evalpoly(x, -1.0000000000000001110, 4.2963016178777127009, -4.0991407924007457612, -6.8442842200833309724, 17.084773793345271001, -13.015133123886661124, 3.9303608629539851049, -0.34636746512247457319) / 
            evalpoly(x, 1.0, -6.6279455994747624059, 17.740962374121397994, -24.446872319343475890, 18.249006287190617068, -7.0580758756624790550, 1.1978786762794003545, -0.053875778140352599789 );
   } else if (z <= -0.18872688282289434049) {          // W >= -2.637, Y_-1
       x=-z/(x0+sqrt(z-z0));
       dwm1c = evalpoly(x, -8.2253155264446844854, -813.20706732001487178, -15270.113237678509000, -79971.585089674149237, -103667.54215808376511, 42284.755505061257427, 74953.525397605484884, 10554.369146366736811) / 
            evalpoly(x, 1.0, 146.36315161669567659, 3912.4761372539240712, 31912.693749754847460, 92441.293717108619527, 94918.733120470346165, 29531.165406571745340, 1641.6808960330370987 );
   } else if (z <= -0.060497597226958343647) {     // W >= -4.253, Y_-2
       x=-z/(x0+sqrt(z-z0));
       dwm1c = evalpoly(x, -9.6184127443354024295, -3557.8569043018004121, -254015.59311284381043, -5.3923893630670639391e6, -3.6638257417536896798e7, -6.1484319486226966213e7, 3.0421690377446134451e7, 3.9728139054879320452e7) / 
            evalpoly(x, 1.0, 507.40525628523300801, 46852.747159777876192, 1.3168304640091436297e6, 1.3111690693712415242e7, 4.6142116445258015195e7, 4.8982268956208830876e7, 9.1959100987983855122e6 );
   } else if (z <= -0.017105334740676008194) {     // W >= -5.832, Y_-3
       x=-z/(x0+sqrt(z-z0));
       dwm1c = evalpoly(x, -11.038489462297466388, -15575.812882656619195, -4.2492947304897773433e6, -3.5170245938803423768e8, -9.8659163036611364640e9, -8.6195372303305003908e10, -1.3286335574027616000e11, 1.5989546434420660462e11) / 
            evalpoly(x, 1.0, 1837.0770693017166818, 612840.97585595092761, 6.2149181398465483037e7, 2.2304011314443083969e9, 2.8254232485273698021e10, 1.0770866639543156165e11, 7.1964698876049131992e10 );
   } else if (z <= -0.0045954962127943706433) {    // W >= -7.382, Y_-4
       x=-z/(x0+sqrt(z-z0));
       dwm1c = evalpoly(x, -12.474405916395746052, -68180.335575543773385, -7.1846599845620093278e7, -2.3142688221759181151e10, -2.5801378337945295130e12, -9.5182748161386314616e13, -8.6073250986210321766e14, 1.4041941853339961439e14) / 
            evalpoly(x, 1.0, 6852.5813734431100971, 8.5153001025466544379e6, 3.2146028239685694655e9, 4.2929807417453196113e11, 2.0234381161638084359e13, 2.8699933268233923842e14, 7.1210136651525477096e14 );
   } else if (z <= -0.0012001610672197724173) {    // W >= -8.913, Y_-5
       x=-z/(x0+sqrt(z-z0));
       dwm1c = evalpoly(x, -13.921651376890072595, -298789.56482388065526, -1.2313019937322092334e9, -1.5556149081899508970e12, -6.8685341106772708734e14, -1.0290616275933266835e17, -4.1404683701619648471e18, -1.4423309998006368397e19) / 
            evalpoly(x, 1.0, 26154.955236499142433, 1.2393087277442041494e8, 1.7832922702470761113e11, 9.0772608163810850446e13, 1.6314734740054252741e16, 8.8371323861233504533e17, 8.4166620643385013384e18 );
   } else if (z <= -0.00030728805932191499844) {   // W >= -10.433, Y_-6
       x=-z/(x0+sqrt(z-z0));
       dwm1c = evalpoly(x, -15.377894224591557534, -1.3122312005096979952e6, -2.1408157022111737888e10, -1.0718287431557811808e14, -1.8849353524027734456e17, -1.1394858607309311995e20, -1.9261555088729141590e22, -3.9978452086676901296e23) / 
            evalpoly(x, 1.0, 101712.86771760620046, 1.8728545945050381188e9, 1.0469617416664402757e13, 2.0704349060120443049e16, 1.4464907902386074496e19, 3.0510432205608900949e21, 1.1397589139790739717e23 );
   } else if (z <= -0.000077447159838062184354) {  // W >= -11.946, Y_-7
       x=-z/(x0+sqrt(z-z0));
       dwm1c = evalpoly(x, -16.841701411264981596, -5.7790823257577138416e6, -3.7757230791256404116e11, -7.5712133742589860941e15, -5.3479338916011465685e19, -1.3082711732297865476e23, -9.1462777004521427440e25, -8.9602768119263629340e27) / 
            evalpoly(x, 1.0, 401820.46666230725328, 2.9211518136900492046e10, 6.4456135373410289079e14, 5.0311809576499530281e18, 1.3879041239716289478e22, 1.1575146167513516225e25, 1.7199220185947756654e27 );
   } else if (z <= -4.5808119698158173174e-17) {     // W >= -41.344, V_-8
       u=log(-z); //? log 
       dwm1c = evalpoly(u, -2.0836260384016439265, 1.6122436242271495710, 5.4464264959637207619, -3.0886331128317160105, 0.46107829155370137880, -0.023553839118456381330, 0.00040538904170253404780, -1.7948156922516825458e-6) / 
            evalpoly(u, 1.0, 2.3699648912703015610, -2.1249449707404812847, 0.38480980098588483913, -0.021720009380176605969, 0.00039405862890608636876, -1.7909312066865957905e-6, 3.1153673308133671452e-12 );
   } else if (z <= -6.1073672236594792982e-79) {     // W >= -185.316, V_-9
       u=log(-z); //? log 
       dwm1c = evalpoly(u, 0.16045383766570541409, 2.2214182524461514029, -0.94119662492050892971, 0.091921523818747869300, -0.0029069760533171663224, 0.000032707247990255961149, -1.2486672336889893018e-7, 1.2247438279861785291e-10) / 
            evalpoly(u, 1.0, -0.70254996087870332289, 0.080974347786703195026, -0.0027469850029563153939, 0.000031943362385183657062, -1.2390620687321666439e-7, 1.2241636115168201999e-10, -1.0275718020546765400e-17 );
   } else if (z < 0.0) {   // V_-10
       u=log(-z); //? log 
       dwm1c = evalpoly(u, -1.2742179703075440564, 1.3696658805421383765, -0.12519345387558783223, 0.0025155722460763844737, -0.000015748033750499977208, 3.4316085386913786410e-8, -2.5025242885340438533e-11, 4.6423885014099583351e-15) / 
            evalpoly(u, 1.0, -0.11420006474152465694, 0.0024285233832122595942, -0.000015520907512751723152, 3.4120534760396002260e-8, -2.4981056186450274587e-11, 4.6419768093059706079e-15, -1.3608713936942602985e-23 );
   } else {
       // throw(DomainError("Expected z < 0.0 but got $z."))
       return -INFINITY;
   }
   return dwm1c;
}

} // namespace detail

// W₀(z) [-1/e,∞) double fast
constexpr double lambertw0_fast(double z) {
    return detail::lambert_dw0c_ep(z + ONE_OVER_E<double>);
}

constexpr double lambertw0(double z) {
    if (z == -ONE_OVER_E<double>) return -1.0;
    return detail::halley_lambertw(detail::lambert_dw0c_ep(z + ONE_OVER_E<double>), z);
}

// W₋₁(z) (-1/e,0) double fast    
constexpr double lambertwm1_fast(double z) {
    return detail::lambert_dwm1c_ep(z, z + ONE_OVER_E<double>);
}

constexpr double lambertwm1(double z) {
    if (z == 0.0) return -INFINITY;
    return detail::halley_lambertw(detail::lambert_dwm1c_ep(z, z + ONE_OVER_E<double>), z);
}

// k-th branch of the Lambert W function. Only accept k = 0 and k = -1 for now!
template<typename T>
constexpr T lambertw(T z,int k=0) { 
    if (k == 0) return lambertw0(z);
    if (k == -1) return lambertwm1(z);
}


namespace detail {
// Fritsch, Shafer, and Crowley (FSC) iteration for womega function
template<typename T>
constexpr auto fsc_womega(T w, T z) {
    T r = z - w - log(w);
    T wp1 = w+1;
    T e = r/wp1*(2*wp1*(wp1+2/3*r)-r)/(2*wp1*(wp1+2/3*r)-2*r);
    return w*(1+e);
}

// Hlley's method for Omega constant, cubic convergence.
template<typename T>
constexpr auto halley_omega(T o) { // 1*exp, 1*div, 5*mul, 7*mul
    T eo = exp(o);
    return (2 + o*(4 + o + eo * o*o)) / (2 + o + eo*(2 + o*(2 + o)));
}
} // namespace detail


// ω(x) float
// Same as womega but using evalpoly() !
constexpr float womega(float x) { 
    //
    //   Single-precision computation of Wright"s omega function, omega(x),
    //   defined as the solution of nonlinear equation, omega(x) + log(omega(x)) = x
    //   by piecewise minimax rational function approximation
    //
    //   Toshio Fukushima <Toshio.Fukushima@nao.ac.jp> 2020/10/17
    //
    //if (isinf(x)) return INFINITY;
    if (x == -INFINITY) return 0;
    if (x == INFINITY) return INFINITY;

    float z, somega;
    if (x < 0.0) {
        if (x > -1.5361328125) {
            somega = evalpoly(x, 0.56714332412913594105 , 0.30630103389199964412, 0.065488479920380545308 , 0.0055153146890067064962 ) / 
                     evalpoly(x, 1.0 , -0.098030204674124009540, 0.048080407489229590984 , -0.0059765940139383360353 );
        } else {
            z=exp(x);
            somega = z*evalpoly(z, 0.99999994780930851871 , 1.7117216658039714225, 0.20020967452709029641 ) /
                     evalpoly(z, 1.0 , 2.7117066932638613716 , 1.4126040690619524729 );
        }
    } else if (x < 2.203125) { 
        somega = evalpoly(x, 0.56714325667175331149 , 0.44204947590429385038, 0.14123153221018897281 , 0.019283157443287762404 ) /
                evalpoly(x, 1.0 , 0.14132449647937415562, 0.028964643040169025625 , -0.00058168896756415762409 );
    } else if (x < 7.6552734375) { 
        somega = evalpoly(x, 0.55937806741718427872 , 0.67956527580178828073, 0.29360195457159048827 , 0.077448100646826501154 ) /
                evalpoly(x, 1.0 , 0.52881376120351465894, 0.081772927185747068186 , -0.000068175768290343786070 );
    } else if (x < 30.7646484375) { 
        somega = evalpoly(x, 0.66104047907992020205 , 0.37877035567803743984, 0.19919433126748008768 ) /
                 evalpoly(x, 1.0 , 0.20879650605789737260, -0.00016947736903018769532 , 2.2611377444850837652e-6, -1.4348602911552556686e-8 );
    } else if (x < 185.2001953125) { 
        somega = evalpoly(x, -0.055532700852464479829 , 0.58620887075870905428, 0.11673091868753091923 , 0.0013099637156785891028 ) / 
                 evalpoly(x, 1.0 , 0.12434745866188878179, 0.0013137744766148976193 , -2.9307591501706925573e-9 );
    } else if (x < 2190.048828125) { 
        somega = evalpoly(x, -2.4215867305897768568 , 0.91134105496756911035, 0.016155964675360530461 , 0.000021533578597524674756 ) / 
                 evalpoly(x, 1.0 , 0.016326356740840015012, 0.000021540941180994856067 , -6.2370181632692660205e-13 );
    } else if (x < 139137.1328125) { 
        somega = evalpoly(x, -5.4363612354490970051 , 0.99287748389227933136, 0.00084052457998910664879 , 4.8894415007562386657e-8 ) / 
                 evalpoly(x, 1.0 , 0.00084106912846565636966, 4.8895005847922485747e-8 , -1.4160730956370852941e-18 );
    } else {
        somega = x*(1.0-log(x) / (x+1.0));   // Single Newton correction applied to asymptotic solution, x
    }
	//? return detail::fsc_womega(somega, x); //TODO: !
    return somega;
}

// ω(x) double
// Same as womega but using evalpoly() !
constexpr double womega(double x) { 
    //
    //   Double-precision computation of Wright"s omega function, omega(x),
    //   defined as the solution of nonlinear equation, omega(x) + log(omega(x)) = x
    //   by piecewise minimax rational function approximation
    //
    //   Toshio Fukushima <Toshio.Fukushima@nao.ac.jp> 2020/10/17
    //
    if (x == -INFINITY) return 0;
    if (x == INFINITY) return INFINITY;

    double z, domega;
    if (x < 0.0) {
        if (x > -1.7969970703125) {
            domega = evalpoly(x, 0.56714329040978393597 , 0.40127273991556628416, 0.16948622586211161962 , 0.046665932688016784538, 0.0085703273322828587102 , 0.0010434401063688186806, 0.000077800083733980925540 + x*2.7204772831680988214e-6 ) /
                     evalpoly(x, 1.0 , 0.069429514456939747074, 0.12462836200029189075 , -0.0038952406229635946107, 0.0044555853359591692983 , -0.00041123146046694083146, 0.000059020981275047793275 , -3.5492274716282929464e-6 );
        } else {
            z = exp(x);
            domega = z*evalpoly(z, 0.9999999999999998931 , 5.0745983953466943659, 8.2606009602900812771 , 4.7024717601199799556, 0.65482227387481846600 ) /
                    evalpoly(z, 1.0 , 6.0745983953465398344, 12.835199355673150217 , 11.092440186086891869, 3.4850593056113482442 , 0.23505959341677892216 );
        }
    } else if (x < 2.6706581115722656250) { 
        domega = evalpoly(x, 0.56714329040978381003 , 0.77356892277028762975, 0.51832380984628428402 , 0.21755702154802737458, 0.061224277815768787167 , 0.011518883525838608343, 0.0013449706921997516673 + x*0.000075896892739201294405 ) /
                 evalpoly(x, 1.0 , 0.72587064520843295454, 0.32082921273199275229 , 0.086948809834337515365, 0.015394071575832887833 , 0.0016360582874079300062, 0.000078307305162627353776 , -3.1442237428129457031e-8 );
    } else if (x < 9.6968069076538085938) { 
        domega = evalpoly(x, 0.56713501343190366161 , 0.78382455352242915730, 0.51756353812338515893 , 0.20726853039949915898, 0.053010998762456477240 , 0.0083925496655250859740, 0.00066092716371736707786 + x*0.000013614908670113335271 ) /
                 evalpoly(x, 1.0 , 0.74389593435342377573, 0.30809335954228387897 , 0.074470884949693883215, 0.010667765269781492069 , 0.00072813578520727560898, 0.000013735475456227982957 , -3.9617637001432374240e-10 );
    } else if (x < 38.007662773132324219) { 
        domega = evalpoly(x, 0.56401353907858227635 , 0.71556555400973446955, 0.40606219668682194772 , 0.13098742100043090882, 0.023792021846516585517 , 0.0014712087559790111411, 0.000028262095549717529699 , 1.3176541606714501011e-7 ) /
                 evalpoly(x, 1.0 , 0.61723923788369990709, 0.19472671474858151447 , 0.029166374057621022340, 0.0016049364440594188061 , 0.000029097481892430916950, 1.3204819260704942813e-7 , -2.2759460679274951500e-13 );
    } else if (x < 224.05800342559814453) { 
        domega = evalpoly(x, 0.74904304655011303763 , 0.41475765300820204606, 0.26229090010420327086 , 0.022231827737742717397, 0.00055769724215842159151 , 4.8929940562358938904e-6, 1.4576036364817152103e-8 , 1.1211845148576067899e-11, 5.8617957261146552764e-19 ) /
                 evalpoly(x, 1.0 , 0.33914957488601671439, 0.024691496687221735503 , 0.00058391033761937493228, 4.9870990455400355673e-6 , 1.4666816425760107778e-8, 1.1216014296054046261e-11 );
    } else if (x < 1318.8293657302856445) { 
        domega = evalpoly(x, -0.52847240530691242510 , 0.66439555408432191153, 0.095734165623936586970 , 0.0017984402717797458483, 9.5608645967684762747e-6 , 1.7130211053307026173e-8, 1.0127406535507852349e-11 , 1.5119473539633986399e-15 ) /
                 evalpoly(x, 1.0 , 0.10436793508415000619, 0.0018544824235611654910 , 9.6789573725809544085e-6, 1.7211790608220386441e-8 , 1.0142134330206435119e-11, 1.5120546649245173808e-15 , -2.8371714954759691899e-24 );
    } else if (x < 9901.8982133865356445) { 
        domega = evalpoly(x, -2.7265755587675828255 , 0.91886110507404644509, 0.015437556964724977333 , 0.000042379757069402316910, 3.3788408654334341745e-8 , 9.0995247456328828154e-12, 8.0541623626962845941e-16 , 1.7892200405606258679e-20 ) /
                 evalpoly(x, 1.0 , 0.015717280527236316462, 0.000042640053213855619617 , 3.3868168872656516608e-8, 9.1075366623274182120e-12 , 8.0562465016652583741e-16, 1.7892388148342752402e-20 , - 7.2802540742878224330e-31 );
    } else if (x < 104729.98977375030518) { 
        domega = evalpoly(x, -4.9758448332804315705 , 0.98672919773901375858, 0.0018048502584507968137 , 5.6727618232662977651e-7, 5.1693477737117469717e-11 , 1.5763106040543895563e-15, 1.5615457093141147610e-20 , 3.8348282655423816844e-26 ) / 
                 evalpoly(x, 1.0 , 0.0018098048079446830033, 5.6778598593151384010e-7 , 5.1710727052210149252e-11, 1.5765001969393981202e-15 , 1.5615988471610678184e-20, 3.8348326592462094894e-26 , -1.8376700258834952892e-38 );
    } else if (x < 1.7873714379758834839e6) { 
        domega = evalpoly(x, -7.5064713916128040209 , 0.99856667809261792330, 0.00014527404521491427728 , 3.6058443746980183631e-9, 2.5487488372192215976e-14 , 5.9071573396622281840e-20, 4.3540463254359589235e-26 , 7.7870710830579392924e-33 ) / 
                 evalpoly(x, 1.0 , 0.00014531469783952414582, 3.6061610745034193796e-9 , 2.5488287561466743981e-14, 5.9072215616613797183e-20 , 4.3540591638996294464e-26, 7.7870717193388587180e-33 , -1.8603410329812104693e-47 );
    } else if (x < 6.4666226364722251892e7) { 
        domega = evalpoly(x, -10.553133866290087103 , 0.99991182205756461588, 6.8147130642100950299e-6 , 7.6772315747426585717e-12, 2.3749238044176130427e-18 , 2.3156775350327431915e-25, 6.8906814806601365039e-33 , 4.7733356492001687570e-41 ) / 
                 evalpoly(x, 1.0 , 6.8148234879449250418e-6, 7.6772685608307803918e-12 , 2.3749276757415131404e-18, 2.3156787740031116785e-25 , 6.8906824237918392311e-33, 4.7733356637141783505e-41 , -1.5212714240163636517e-58 );
    } else if (x < 9.6537874592561578751e9) { 
        domega = evalpoly(x, -14.453233847636304145 , 0.99999773787655695790, 1.3384094485652721805e-7 , 2.7649127536426257913e-15, 1.4462424184214411200e-23 , 2.1739442557543959836e-32, 9.0218885020989014506e-42 , 7.8687375038593681713e-52 ) / 
                 evalpoly(x, 1.0 , 1.3384099578765282610e-7, 2.7649130385236672771e-15 , 1.4462424639425969762e-23, 2.1739442758770299520e-32 , 9.0218885211281218582e-42, 7.8687375041325222089e-52 , -2.9925825575484892462e-73 );
    } else if (x < 3.7284375517631785583e11) { 
        domega = evalpoly(x, -19.911981551589239565 , 0.99999998802702964239, 5.3193290756933022976e-10 , 3.6315161298281125840e-20, 4.6372938690029442720e-31 , 9.2610561378209620950e-43 ) / 
                 evalpoly(x, 1.0 , 5.3193290845021667622e-10, 3.6315161310308030589e-20 , 4.6372938692633433675e-31, 9.2610561378259071409e-43 );
    } else {
        domega = x*(1.0-log(x)/(x+1.0));   // Single Newton correction applied to asymptotic solution, x
    }
	//? return detail::fsc_womega(somega, x); //TODO: !
    return domega;
}


} // namespace cx


/// TESTS 
/// TESTS 
/// TESTS 

namespace cx::tests {

static_assert(leading_zeros(1u) == 31);
static_assert(trailing_zeros(2u) == 1);

static_assert(leading_ones(uint32_t(ipow(2u,32) - 2)) == 31);
static_assert(trailing_ones(3u) == 2);

static_assert(gcd(6, 9) == 3);
static_assert(gcd(6, -9) == 3);
static_assert(gcd(0, 0, 10, 15) == 5);

static_assert(lcm(2, 3) == 6);
static_assert(lcm(-2, 3) == 6);
static_assert(lcm(0, 3) == 0);
static_assert(lcm(0, 0) == 0);

static_assert(lcm(1, 3, 5, 7) == 105);
static_assert(lcm(2*3*4, 3*4*5, 4*5*6, 5*6*7) == 840);

static_assert(agm(2.0, 3.0) > 2.47);
static_assert(agm(10.0, 20.0) > 14.56);

static_assert(is_zero( 0.0) == true);
static_assert(is_zero( 0.0f) == true);
static_assert(is_zero(-0.0) == true);
static_assert(is_zero(-0.0f) == true);
static_assert(is_zero(-10.0) == false);
static_assert(is_zero(-10.0f) == false);


static_assert(isnan(double(NAN)) == true);
static_assert(isnan(float(NAN)) == true);
static_assert(isnan(-10.0) == false);
static_assert(isnan(-10.0f) == false);


static_assert(isinf(-10.0) == false);
static_assert(isinf(-10.0f) == false);
static_assert(isinf(double(INFINITY)) == true);
static_assert(isinf(float(INFINITY)) == true);

static_assert(detail::is_finite(-10.0));
static_assert(detail::is_finite(-10.0f));

static_assert(detail::round(102030.458) == 102030.0);
static_assert(detail::round(102030.589) == 102031.0);

static_assert(detail::roundi(102030.123) == 102030);
static_assert(detail::roundi(102030.589) == 102031);

//static_assert(detail::bit_or(true, 1.0f));
//static_assert(detail::bit_or(true, 1.0));

static_assert(detail::bit_or(-0.0f, 1.0f) == -1.0f);
static_assert(detail::bit_or(-0.0, 1.0) == -1.0);


static_assert(if_add(false, 5.0f, 3.0f) == 5.0f);
static_assert(if_add(true, 5.0f, 3.0f) == 8.0f);

static_assert(if_add(false, 5.0, 3.0 ) == 5.0);
static_assert(if_add(true, 5.0, 3.0 ) == 8.0);

static_assert(abs(-10.5f) == 10.5f);
static_assert(abs(-10.5) == 10.5);

static_assert(min(2.0,1.0,3.0) == 1.0);
static_assert(max(2.0,1.0,3.0) == 3.0);

static_assert(and_all(true));
static_assert(and_all(true, true, true));
static_assert(!and_all(false, true, true));

static_assert(or_all(true));
static_assert(or_all(false, false, true));
static_assert(!or_all(false, false, false));

//? static_assert(sum() == 0.0);
static_assert(sum(2.0) == 2.0);
static_assert(sum(2.0,2.0,2.0,2.0) == 8.0);
static_assert(product(2.0,3.0,4.0,5.0,6.0) == factorial(6));
static_assert(product(2.0,2.0,2.0) == pow(2.0,3.0));



static_assert(cx::copysign(1.0f, -8.8f) == -1.0f);
static_assert(cx::copysign(1.0, -8.8) == -1.0);

static_assert(cx::copysign(-21.2f, 8.0f) == 21.2f);
static_assert(cx::copysign(-21.2, 8.8) == 21.2);

static_assert(cx::fabs(-3.3) == 3.3);
static_assert(cx::fabs(-3.3f) == 3.3f);

static_assert(cx::abs(0) == 0);
static_assert(cx::abs(-33) == 33);
static_assert(cx::abs(33u) == 33u);


static_assert(cx::floor(1.0) == 1.0);
static_assert(cx::floor(1.1) == 1.0);
static_assert(cx::floor(1.6) == 1.0);
static_assert(cx::floor(-1.1) == -2.0);
static_assert(cx::floor(-1.6) == -2.0);

static_assert(cx::ceil(1.0) == 1.0);
static_assert(cx::ceil(1.1) == 2.0);
static_assert(cx::ceil(1.6) == 2.0);
static_assert(cx::ceil(-1.1) == -1.0);
static_assert(cx::ceil(-1.6) == -1.0);

static_assert(cx::round(-5.5) == -6.0);
static_assert(cx::floor(-5.5) == -6.0);
static_assert(cx::ceil(-5.5) == -5.0);
static_assert(cx::trunc(-5.5) == -5.0);

static_assert(cx::round(2.3) == 2.0);
static_assert(cx::round(-3.8) == -4.0);

static_assert(cx::trunc(2.3) == 2.0);
static_assert(cx::trunc(-3.8) == -3.0);

static_assert(cx::round(5.5f) == 6.0f);
static_assert(cx::round(-5.5f) == -6.0f);
static_assert(cx::round(-4.5f) == -5.0f);
static_assert(cx::round(-3.5f) == -4.0f);
static_assert(cx::round(-2.5f) == -3.0f);

static_assert(cx::floor(-5.5f) == -6.0f);
static_assert(cx::ceil(-5.5f) == -5.0f);
static_assert(cx::trunc(-5.5f) == -5.0f);

static_assert(cx::round(2.3f) == 2.0f);
static_assert(cx::round(-3.8f) == -4.0f);

static_assert(cx::trunc(2.3f) == 2.0f);
static_assert(cx::trunc(-3.8f) == -3.0f);



static_assert(eps(1.0f) == std::numeric_limits<float>::epsilon());
static_assert(eps(1.0) == std::numeric_limits<double>::epsilon());


// static_assert(cx::sqrtf(4.0f) == 2.0f);
//static_assert(cx::sqrtf(0.0f) == 0.0f);
//static_assert(sqrtf(0.0f) == 0.0f);


static_assert(cx::sqrt(4.0) == 2.0);
static_assert(cx::sqrt(0.0) == 0.0);
static_assert(sqrt(0.0) == 0.0);



// static_assert(sqrtf(4.0f) == 2.0f);
static_assert(sqrt(4.0) == 2.0);


static_assert(cbrtf(0.0f) == 0.0f);
//! static_assert(cbrtf(-8.0f) == -2.0f);

static_assert(cbrt(0.0) == 0.0);
static_assert(cbrt(-8.0) == -2.0);

static_assert(hypot(3.0f,4.0f) == 5.0f);
static_assert(hypot(3.0,4.0) == 5.0);
static_assert(hypot(1.0e21f, 0.0f) == 1.0e21f);
static_assert(hypot(1.0e200, 0.0) == 1.0e200);

static_assert(hypot(3.0,4.0,0.0) == 5.0);
static_assert(hypot(1.0e21f, 0.0f, 0.0f) == 1.0e21f);

static_assert(hypot(1.0, 0.0, 0.0) == 1.0f);
static_assert(hypot(1.0f, 0.0f, 0.0f, 0.0f) == 1.0f);

static_assert(hypot(1.0e200, 0.0, 0.0) == 1.0e200);


static_assert(isqrt(0) == 0);
static_assert(isqrt(1) == 1);
static_assert(isqrt(2) == 1);
static_assert(isqrt(4) == 2);
static_assert(isqrt(9) == 3);
static_assert(isqrt(25) == 5);
static_assert(isqrt(100) == 10);
static_assert(isqrt(225) == 15);
static_assert(isqrt(65535u * 65535u) == 65535); // uint32_t
static_assert(isqrt(4294967295llu * 4294967295llu) == 4294967295llu); // uint64_t

static_assert(isqrt(ipow(2llu,62)) == ipow(2llu,31));

static_assert(sqrt(0.0) == 0.0);
static_assert(sqrt(1.0) == 1.0);
static_assert(sqrt(4.0) == 2.0);
static_assert(sqrt(25.0) == 5.0);
static_assert(sqrt(100.0) == 10.0);
static_assert(sqrt(1000000.0) == 1000.0);
static_assert(sqrt(1000000000000.0) == 1000000.0);
static_assert(sqrt(1.0e20) == 1.0e10);
static_assert(sqrt(1.0e100) == 1.0e50);
// static_assert(sqrt(1.0e200) == 1.0e100);

static_assert(sqrt(0.0f) == 0.0f);
static_assert(sqrt(1.0f) == 1.0f);
static_assert(sqrt(4.0f) == 2.0f);
static_assert(sqrt(25.0f) == 5.0f);
static_assert(sqrt(100.0f) == 10.0f);
static_assert(sqrt(1000000.0f) == 1000.0f);
static_assert(sqrt(1000000000000.0f) == 1000000.0f);
static_assert(sqrt(1.0e18f) == 1.0e9f);
// static_assert(sqrt(1.0e20f) == 1.0e10f);

//static_assert(rsqrt(0.0) == 0.0);
static_assert(rsqrt(1.0) == 1.0);
static_assert(rsqrt(4.0) == 1.0/2.0);
static_assert(rsqrt(16.0) == 1.0/4.0); 
static_assert(rsqrt(25.0) <= 1.0/5.0); // TODO: !
static_assert(rsqrt(100.0) <= 1.0/10.0); // TODO: !

//static_assert(rsqrt(0.0f) == 0.0f);
static_assert(rsqrt(1.0f) == 1.0f); 
static_assert(rsqrt(4.0f) == 1.0f/2.0f);
static_assert(rsqrt(16.0f) == 1.0f/4.0f); 
static_assert(rsqrt(25.0f) <= 1.0f/5.0f); // TODO: !
static_assert(rsqrt(100.0f) <= 1.0f/10.0f); // TODO: !

static_assert(ipow(1, 10) == 1);
static_assert(ipow(2, 4) == 16);

static_assert(is_pow2(2));
static_assert(is_pow2(4));
static_assert(is_pow2(256));
static_assert(is_pow2(4294967296u));
static_assert(is_pow2(4611686018427387904llu));


static_assert(powf(1.0f, 10.0f) == 1.0f);
static_assert(powf(2.0f, 4.0f) == 16.0f);
// static_assert(powf(4.0f, 0.5f) == 2.0f);

static_assert(pow(1.0, 10.0) == 1.0);
static_assert(pow(2.0, 4.0) == 16.0);
// static_assert(powf(4.0, 0.5) == 2.0);

static_assert(vm_pow(1.0f, 10.0f) == 1.0f);
static_assert(vm_pow(2.0f, 4.0f) == 16.0f);

//static_assert(vm_pow(1.0, 10.0) == 0.0); //! DEBUG 

static_assert(vm_pow(0.0, 10.0) == 0.0);
static_assert(vm_pow(-0.0, 10.0) == -0.0);

static_assert(vm_pow(-1.0, 10.0) == 1.0);
static_assert(vm_pow(-1.0, 9.0) == -1.0);

static_assert(vm_pow(2.0, 4.0) == 16.0);

static_assert(vm_pow(2.0, -2.0) == 0.25);


static_assert(log(1.0f) == 0.0f);
static_assert(log(1.0) == 0.0);

static_assert(log(125.0) / log(5.0) == 3.0000000000000004);

static_assert(log3(27.0) == 3.0);
//static_assert(log(3.0, 27.0) == 3.0);

static_assert(log10(10.0) == 1.0);
static_assert(log10(100.0) == 2.0);
static_assert(log10(1000.0) == 3.0);
static_assert(log10(10000.0) == 4.0);

static_assert(log(10.0,10000.0) == 4.0);

// static_assert(exp(1.0) == 2.718281828459045);
static_assert(exp(1.0) == 0xa.df85458a2bb5p-2);


// sin double 
static_assert(sin(0.0) == 0.0);
static_assert(sin(PI<double>/2.0) == 1.0);
static_assert(sin(PI<double>) <= DBL_EPSILON);

// cos double 
static_assert(cos(0.0) == 1.0);
static_assert(cos(PI<double>/2.0) <= DBL_EPSILON);
static_assert(cos(PI<double>) == -1.0);
static_assert(cos(0.7390851332151607) == 0.7390851332151607);

// tan double 
static_assert(tan(0.0) == 0.0);
static_assert(tan(PI<double>) <= DBL_EPSILON);


// sin float 
static_assert(sin(0.0f) == 0.0f);
static_assert(sin(PI<float> / 2.0f) == 1.0f);
static_assert(sin(PI<float>) <= FLT_EPSILON);

// cos float 
static_assert(cos(0.0f) == 1.0f);
static_assert(cos(PI<float>/2.0f) <= FLT_EPSILON);
static_assert(cos(PI<float>) == -1.0f);

// tan float 
static_assert(tan(0.0f) == 0.0f);
static_assert(tan(PI<float>) <= FLT_EPSILON);


static_assert(atan2(0.0, 0.0) == 0.0);
static_assert(atan2(0.0f, 0.0f) == 0.0f);

static_assert(is_prime(2));
static_assert(is_prime(3));
static_assert(is_prime(5));
static_assert(is_prime(7));
static_assert(is_prime(11));
static_assert(is_prime(13));
static_assert(is_prime(17));
static_assert(is_prime(19));
static_assert(is_prime(23));
static_assert(is_prime(101));

static_assert(!is_prime(4));
static_assert(!is_prime(9));

static_assert(!is_prime(11 * 11));
static_assert(!is_prime(13 * 13));
static_assert(!is_prime(17 * 17));
static_assert(!is_prime(19 * 19));
static_assert(!is_prime(23 * 23));

static_assert(!is_prime(16));
static_assert(!is_prime(25));
static_assert(!is_prime(36));

static_assert(is_prime(4294967311));

static_assert(is_prime((193)));
static_assert(is_prime((407521)));
static_assert(is_prime((299210837)));

static_assert(is_prime(uint64_t(193)));
static_assert(is_prime(uint64_t(407521)));
static_assert(is_prime(uint64_t(299210837)));

//static_assert(is_prime_slow(4294967311));
//static_assert(is_prime_slow(720575940379));

static_assert(is_prime(720575940379));
static_assert(!is_prime(720575940380));

static_assert(is_prime(4611686018427387847llu));
static_assert(!is_prime(4611686018427387848llu));

static_assert(is_prime(9223372036864775767llu));

static_assert(all_primes(
2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,
163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,
337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503));

/*static_assert(all_primes(
	720575940379, 720575940401, 720575940407, 720575940421, 720575940431, 720575940437,
	720575940439, 720575940449, 720575940457, 720575940463, 720575940509, 720575940509));*/

static_assert(!is_prime(ipow(2llu,63)));
static_assert(ipow(2llu, 63) == 9223372036854775808llu);
//static_assert(ipow(2llu, 64) == 18446744073709551616llu);
//static_assert(ipow(2, 63) == 9223372036854775808);

static_assert(is_odd(3u));
static_assert(is_even(2u));

static_assert(all_odd(1,3,5));
static_assert(all_even(0,2,4));

/// solve_cubic
static_assert(solve_cubic(1.0,1.0,1.0,0.0).n == 1);
static_assert(solve_cubic(1.0,1.0,1.0,0.0).x[0] == 0.0);

static_assert(solve_cubic(1.0,1.0,1.0,1.0).n == 1);
static_assert(almost_equal_ulps(solve_cubic(1.0,1.0,1.0,1.0).x[0],-1.0));

static_assert(solve_cubic(2.0,-3.0,4.0,-6.0).n == 1);
static_assert(solve_cubic(2.0,-3.0,4.0,-6.0).x[0] == 1.5);

static_assert(solve_cubic(-1.0,3.0,4.0,-6.0).n == 3);
static_assert(almost_equal_ulps(solve_cubic(-1.0,3.0,4.0,-6.0).x[2], 1.0, 3u));

//? static_assert(gamma(10.0) == 362880.0);
//? static_assert(gamma(10.0f) == 362880.0f);

static_assert(cx::lambertw0(0.0) == 0.0);
static_assert(cx::lambertw0(1.0) == cx::womega(0.0));
static_assert(cx::womega(1.0) == 1.0);
static_assert(cx::lambertwm1(-1.0/cx::E<double>) == -1.0);

} // namespace cx::tests