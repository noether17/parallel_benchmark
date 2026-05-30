#pragma once

template <typename Operation>
void synchronize(Operation&&) {}

template <typename AsyncOperation>
void synchronize(AsyncOperation&& ao)
  requires requires(AsyncOperation ao) { ao.synchronize(); }
{
  ao.synchronize();
}
