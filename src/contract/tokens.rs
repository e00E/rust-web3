//! Mapping of rust types to [`Token`](ethabi::Token) for easier calling of solidity functions from
//! rust.
//!
//! `SingleTokenize` is implemented for types that map to a single `Token`. For example leaf types
//! like `i32` and `String` or compound types of leaf types like `[i32]` and `(i32, i32)`.
//!
//! `Tokenize` is implemented for tuples of `Tokenize`. These tuples represent a group of arguments
//! or return types that are used for when ethabi's encode and decode functions work with
//! `Vec<Token>`.
//!
//! ```
//! use ethabi::Token;
//! use web3::contract::tokens::{MultiTokenize, SingleTokenize};
//!
//! // We have a rust tuple
//! let tuple = (false, true);
//! // and the equivalent tokens.
//! let tokens = vec![Token::Bool(false), Token::Bool(true)];
//!
//! // When treating the tuple as a single a token it becomes `Token::Tuple`
//! assert_eq!(tuple.into_token(), Token::Tuple(tokens.clone()));
//! // and when treating it as a collection of arguments it becomes a vector of two bool tokens.
//! assert_eq!(tuple.into_tokens(), tokens.clone());
//! // If we wanted to treat it as a single tuple argument we would wrap it in another tuple.
//! assert_eq!((tuple,).into_tokens(), vec![Token::Tuple(tokens.clone())]);
//! ```

use crate::{
    contract::error::Error,
    types::{Address, Bytes, BytesArray, H256, U128, U256},
};
use arrayvec::ArrayVec;
use ethabi::Token;

/// Tokenization to and from a single token.
pub trait SingleTokenize {
    /// Convert Token into Self.
    fn from_token(token: Token) -> Result<Self, Error>
    where
        Self: Sized;
    /// Convert Self into Token.
    fn into_token(self) -> Token;
}

impl SingleTokenize for Token {
    fn from_token(token: Token) -> Result<Self, Error> {
        Ok(token)
    }
    fn into_token(self) -> Token {
        self
    }
}

impl SingleTokenize for String {
    fn from_token(token: Token) -> Result<Self, Error> {
        match token {
            Token::String(s) => Ok(s),
            other => Err(Error::InvalidOutputType(format!("Expected `String`, got {:?}", other))),
        }
    }

    fn into_token(self) -> Token {
        Token::String(self)
    }
}

impl SingleTokenize for Bytes {
    fn from_token(token: Token) -> Result<Self, Error> {
        match token {
            Token::Bytes(s) => Ok(s.into()),
            other => Err(Error::InvalidOutputType(format!("Expected `Bytes`, got {:?}", other))),
        }
    }

    fn into_token(self) -> Token {
        Token::Bytes(self.0)
    }
}

impl SingleTokenize for H256 {
    fn from_token(token: Token) -> Result<Self, Error> {
        match token {
            Token::FixedBytes(mut s) => {
                if s.len() != 32 {
                    return Err(Error::InvalidOutputType(format!("Expected `H256`, got {:?}", s)));
                }
                let mut data = [0; 32];
                for (idx, val) in s.drain(..).enumerate() {
                    data[idx] = val;
                }
                Ok(data.into())
            }
            other => Err(Error::InvalidOutputType(format!("Expected `H256`, got {:?}", other))),
        }
    }

    fn into_token(self) -> Token {
        Token::FixedBytes(self.as_ref().to_vec())
    }
}

impl SingleTokenize for Address {
    fn from_token(token: Token) -> Result<Self, Error> {
        match token {
            Token::Address(data) => Ok(data),
            other => Err(Error::InvalidOutputType(format!("Expected `Address`, got {:?}", other))),
        }
    }

    fn into_token(self) -> Token {
        Token::Address(self)
    }
}

macro_rules! eth_uint_tokenizable {
    ($uint: ident, $name: expr) => {
        impl SingleTokenize for $uint {
            fn from_token(token: Token) -> Result<Self, Error> {
                match token {
                    Token::Int(data) | Token::Uint(data) => Ok(::std::convert::TryInto::try_into(data).unwrap()),
                    other => Err(Error::InvalidOutputType(format!("Expected `{}`, got {:?}", $name, other)).into()),
                }
            }

            fn into_token(self) -> Token {
                Token::Uint(self.into())
            }
        }
    };
}

eth_uint_tokenizable!(U256, "U256");
eth_uint_tokenizable!(U128, "U128");

macro_rules! int_tokenizable {
    ($int: ident, $token: ident) => {
        impl SingleTokenize for $int {
            fn from_token(token: Token) -> Result<Self, Error> {
                match token {
                    Token::Int(data) | Token::Uint(data) => Ok(data.low_u128() as _),
                    other => Err(Error::InvalidOutputType(format!(
                        "Expected `{}`, got {:?}",
                        stringify!($int),
                        other
                    ))),
                }
            }

            fn into_token(self) -> Token {
                // this should get optimized away by the compiler for unsigned integers
                #[allow(unused_comparisons)]
                let data = if self < 0 {
                    // NOTE: Rust does sign extension when converting from a
                    // signed integer to an unsigned integer, so:
                    // `-1u8 as u128 == u128::max_value()`
                    U256::from(self as u128) | U256([0, 0, u64::max_value(), u64::max_value()])
                } else {
                    self.into()
                };
                Token::$token(data)
            }
        }
    };
}

int_tokenizable!(i8, Int);
int_tokenizable!(i16, Int);
int_tokenizable!(i32, Int);
int_tokenizable!(i64, Int);
int_tokenizable!(i128, Int);
int_tokenizable!(u8, Uint);
int_tokenizable!(u16, Uint);
int_tokenizable!(u32, Uint);
int_tokenizable!(u64, Uint);
int_tokenizable!(u128, Uint);

impl SingleTokenize for bool {
    fn from_token(token: Token) -> Result<Self, Error> {
        match token {
            Token::Bool(data) => Ok(data),
            other => Err(Error::InvalidOutputType(format!("Expected `bool`, got {:?}", other))),
        }
    }
    fn into_token(self) -> Token {
        Token::Bool(self)
    }
}

macro_rules! impl_single_tokenize {
    ($count: expr, $( $ty: ident : $no: tt, )*) => {
        impl<$($ty, )*> SingleTokenize for ($($ty,)*)
        where
            $($ty: SingleTokenize,)*
        {
            fn from_token(token: Token) -> Result<Self, Error>
            {
                let mut tokens = match token {
                    Token::Tuple(tokens) => tokens,
                    _ => return Err(Error::InvalidOutputType(format!("expected tuple"))),
                };
                if tokens.len() != $count {
                    return Err(Error::InvalidOutputType(format!("expected tuple of size {} but got {}", $count, tokens.len())));
                }
                #[allow(unused_variables)]
                #[allow(unused_mut)]
                let mut drain = tokens.drain(..);
                Ok(($($ty::from_token(drain.next().unwrap())?,)*))
            }

            fn into_token(self) -> Token {
                Token::Tuple(vec![$(self.$no.into_token(),)*])
            }
        }
    }
}

impl_single_tokenize!(0,);
impl_single_tokenize!(1, A:0, );
impl_single_tokenize!(2, A:0, B:1, );
impl_single_tokenize!(3, A:0, B:1, C:2, );
impl_single_tokenize!(4, A:0, B:1, C:2, D:3, );
impl_single_tokenize!(5, A:0, B:1, C:2, D:3, E:4, );
impl_single_tokenize!(6, A:0, B:1, C:2, D:3, E:4, F:5, );
impl_single_tokenize!(7, A:0, B:1, C:2, D:3, E:4, F:5, G:6, );
impl_single_tokenize!(8, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, );
impl_single_tokenize!(9, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, );
impl_single_tokenize!(10, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, );
impl_single_tokenize!(11, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, );
impl_single_tokenize!(12, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, );
impl_single_tokenize!(13, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, );
impl_single_tokenize!(14, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, );
impl_single_tokenize!(15, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14, );
impl_single_tokenize!(16, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14, P:15, );

/// Marker trait for `Tokenize` types that can be tokenized to and from a
/// `Token::Array` and `Token:FixedArray`.
pub trait TokenizableItem: SingleTokenize {}

macro_rules! tokenizable_item {
    ($($type: ty,)*) => {
        $(
            impl TokenizableItem for $type {}
        )*
    };
}

tokenizable_item! {
    Token, String, Address, H256, U256, U128, bool, BytesArray, Vec<u8>,
    i8, i16, i32, i64, i128, u16, u32, u64, u128,
}

impl SingleTokenize for BytesArray {
    fn from_token(token: Token) -> Result<Self, Error> {
        match token {
            Token::FixedArray(tokens) | Token::Array(tokens) => {
                let bytes = tokens
                    .into_iter()
                    .map(SingleTokenize::from_token)
                    .collect::<Result<Vec<u8>, Error>>()?;
                Ok(Self(bytes))
            }
            other => Err(Error::InvalidOutputType(format!("Expected `Array`, got {:?}", other))),
        }
    }

    fn into_token(self) -> Token {
        Token::Array(self.0.into_iter().map(SingleTokenize::into_token).collect())
    }
}

impl SingleTokenize for Vec<u8> {
    fn from_token(token: Token) -> Result<Self, Error> {
        match token {
            Token::Bytes(data) => Ok(data),
            Token::FixedBytes(data) => Ok(data),
            other => Err(Error::InvalidOutputType(format!("Expected `bytes`, got {:?}", other))),
        }
    }
    fn into_token(self) -> Token {
        Token::Bytes(self)
    }
}

impl<T: TokenizableItem> SingleTokenize for Vec<T> {
    fn from_token(token: Token) -> Result<Self, Error> {
        match token {
            Token::FixedArray(tokens) | Token::Array(tokens) => {
                tokens.into_iter().map(SingleTokenize::from_token).collect()
            }
            other => Err(Error::InvalidOutputType(format!("Expected `Array`, got {:?}", other))),
        }
    }

    fn into_token(self) -> Token {
        Token::Array(self.into_iter().map(SingleTokenize::into_token).collect())
    }
}

impl<T: TokenizableItem> TokenizableItem for Vec<T> {}

macro_rules! impl_fixed_types {
    ($num: expr) => {
        impl SingleTokenize for [u8; $num] {
            fn from_token(token: Token) -> Result<Self, Error> {
                match token {
                    Token::FixedBytes(bytes) => {
                        if bytes.len() != $num {
                            return Err(Error::InvalidOutputType(format!(
                                "Expected `FixedBytes({})`, got FixedBytes({})",
                                $num,
                                bytes.len()
                            )));
                        }

                        let mut arr = [0; $num];
                        arr.copy_from_slice(&bytes);
                        Ok(arr)
                    }
                    other => Err(
                        Error::InvalidOutputType(format!("Expected `FixedBytes({})`, got {:?}", $num, other)).into(),
                    ),
                }
            }

            fn into_token(self) -> Token {
                Token::FixedBytes(self.to_vec())
            }
        }

        impl TokenizableItem for [u8; $num] {}

        impl<T: TokenizableItem + Clone> SingleTokenize for [T; $num] {
            fn from_token(token: Token) -> Result<Self, Error> {
                match token {
                    Token::FixedArray(tokens) => {
                        if tokens.len() != $num {
                            return Err(Error::InvalidOutputType(format!(
                                "Expected `FixedArray({})`, got FixedArray({})",
                                $num,
                                tokens.len()
                            )));
                        }

                        let mut arr = ArrayVec::<[T; $num]>::new();
                        let mut it = tokens.into_iter().map(T::from_token);
                        for _ in 0..$num {
                            arr.push(it.next().expect("Length validated in guard; qed")?);
                        }
                        // Can't use expect here because [T; $num]: Debug is not satisfied.
                        match arr.into_inner() {
                            Ok(arr) => Ok(arr),
                            Err(_) => panic!("All elements inserted so the array is full; qed"),
                        }
                    }
                    other => Err(
                        Error::InvalidOutputType(format!("Expected `FixedArray({})`, got {:?}", $num, other)).into(),
                    ),
                }
            }

            fn into_token(self) -> Token {
                Token::FixedArray(ArrayVec::from(self).into_iter().map(T::into_token).collect())
            }
        }

        impl<T: TokenizableItem + Clone> TokenizableItem for [T; $num] {}
    };
}

impl_fixed_types!(1);
impl_fixed_types!(2);
impl_fixed_types!(3);
impl_fixed_types!(4);
impl_fixed_types!(5);
impl_fixed_types!(6);
impl_fixed_types!(7);
impl_fixed_types!(8);
impl_fixed_types!(9);
impl_fixed_types!(10);
impl_fixed_types!(11);
impl_fixed_types!(12);
impl_fixed_types!(13);
impl_fixed_types!(14);
impl_fixed_types!(15);
impl_fixed_types!(16);
impl_fixed_types!(32);
impl_fixed_types!(64);
impl_fixed_types!(128);
impl_fixed_types!(256);
impl_fixed_types!(512);
impl_fixed_types!(1024);

/// Tokenization to and from multiple tokens used for function arguments and return values.
pub trait MultiTokenize {
    /// Convert Tokens into Self.
    fn from_tokens(tokens: Vec<Token>) -> Result<Self, Error>
    where
        Self: Sized;
    /// Convert Self into Tokens
    fn into_tokens(self) -> Vec<Token>;
}

macro_rules! impl_multi_tokenize {
    ($count: expr, $( $ty: ident : $no: tt, )*) => {
        impl<$($ty, )*> MultiTokenize for ($($ty,)*)
        where
            $($ty: SingleTokenize,)*
        {
            fn from_tokens(mut tokens: Vec<Token>) -> Result<Self, Error>
            {
                if tokens.len() != $count {
                    return Err(Error::InvalidOutputType(format!("expected {} tokens but got {}", $count, tokens.len())));
                }
                #[allow(unused_variables)]
                #[allow(unused_mut)]
                let mut drain = tokens.drain(..);
                Ok(($($ty::from_token(drain.next().unwrap())?,)*))
            }

            fn into_tokens(self) -> Vec<Token> {
                vec![$(self.$no.into_token(),)*]
            }
        }
    }
}

impl_multi_tokenize!(0,);
impl_multi_tokenize!(1, A:0, );
impl_multi_tokenize!(2, A:0, B:1, );
impl_multi_tokenize!(3, A:0, B:1, C:2, );
impl_multi_tokenize!(4, A:0, B:1, C:2, D:3, );
impl_multi_tokenize!(5, A:0, B:1, C:2, D:3, E:4, );
impl_multi_tokenize!(6, A:0, B:1, C:2, D:3, E:4, F:5, );
impl_multi_tokenize!(7, A:0, B:1, C:2, D:3, E:4, F:5, G:6, );
impl_multi_tokenize!(8, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, );
impl_multi_tokenize!(9, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, );
impl_multi_tokenize!(10, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, );
impl_multi_tokenize!(11, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, );
impl_multi_tokenize!(12, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, );
impl_multi_tokenize!(13, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, );
impl_multi_tokenize!(14, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, );
impl_multi_tokenize!(15, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14, );
impl_multi_tokenize!(16, A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14, P:15, );

impl MultiTokenize for Vec<Token> {
    fn from_tokens(tokens: Vec<Token>) -> Result<Self, Error>
    where
        Self: Sized,
    {
        Ok(tokens)
    }

    fn into_tokens(self) -> Vec<Token> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Address, BytesArray, U256};
    use ethabi::{Token, Uint};
    use hex_literal::hex;

    fn assert_tokenizable<T: SingleTokenize>() {}
    fn assert_multi_tokenizable<T: MultiTokenize>() {}

    #[test]
    #[ignore]
    fn should_compile() {
        assert_tokenizable::<Vec<Token>>();
        assert_tokenizable::<U256>();
        assert_tokenizable::<Address>();
        assert_tokenizable::<String>();
        assert_tokenizable::<bool>();
        assert_tokenizable::<Vec<u8>>();
        assert_tokenizable::<BytesArray>();
        assert_tokenizable::<(U256, bool)>();
        assert_multi_tokenizable::<(U256, bool)>();
        assert_tokenizable::<Vec<U256>>();
        assert_tokenizable::<[U256; 4]>();
        assert_tokenizable::<Vec<[[u8; 1]; 64]>>();
        assert_tokenizable::<(Vec<Vec<u8>>, [U256; 4], Vec<U256>, U256)>();
        assert_tokenizable::<(i8, i16, i32, i64, i128)>();
        assert_multi_tokenizable::<(i8, i16, i32, i64, i128)>();
        assert_tokenizable::<(u16, u32, u64, u128)>();
        assert_multi_tokenizable::<(u16, u32, u64, u128)>();
        assert_multi_tokenizable::<Vec<Token>>();
    }

    #[test]
    fn nested_tuples() {
        type T = (u8, (u16, (u32, u64)));
        let tuple: T = (0, (1, (2, 3)));
        let expected = vec![
            Token::Uint(0.into()),
            Token::Tuple(vec![
                Token::Uint(1.into()),
                Token::Tuple(vec![Token::Uint(2.into()), Token::Uint(3.into())]),
            ]),
        ];
        assert_eq!(tuple.into_token(), Token::Tuple(expected.clone()));
        assert_eq!(T::from_token(Token::Tuple(expected.clone())).unwrap(), tuple);
        assert_eq!(tuple.into_tokens(), expected);
        assert_eq!(T::from_tokens(expected).unwrap(), tuple);
    }

    #[test]
    fn should_decode_array_of_fixed_bytes() {
        // byte[8][]
        let tokens = Token::FixedArray(vec![
            Token::FixedBytes(hex!("01").into()),
            Token::FixedBytes(hex!("02").into()),
            Token::FixedBytes(hex!("03").into()),
            Token::FixedBytes(hex!("04").into()),
            Token::FixedBytes(hex!("05").into()),
            Token::FixedBytes(hex!("06").into()),
            Token::FixedBytes(hex!("07").into()),
            Token::FixedBytes(hex!("08").into()),
        ]);
        let data: [[u8; 1]; 8] = SingleTokenize::from_token(tokens).unwrap();
        assert_eq!(data[0][0], 1);
        assert_eq!(data[1][0], 2);
        assert_eq!(data[2][0], 3);
        assert_eq!(data[7][0], 8);
    }

    #[test]
    fn should_decode_array_of_bytes() {
        let token = Token::Array(vec![Token::Uint(Uint::from(0)), Token::Uint(Uint::from(1))]);
        let data: BytesArray = SingleTokenize::from_token(token).unwrap();
        assert_eq!(data.0[0], 0);
        assert_eq!(data.0[1], 1);
    }

    #[test]
    fn should_sign_extend_negative_integers() {
        assert_eq!((-1i8).into_token(), Token::Int(U256::MAX));
        assert_eq!((-2i16).into_token(), Token::Int(U256::MAX - 1));
        assert_eq!((-3i32).into_token(), Token::Int(U256::MAX - 2));
        assert_eq!((-4i64).into_token(), Token::Int(U256::MAX - 3));
        assert_eq!((-5i128).into_token(), Token::Int(U256::MAX - 4));
    }
}
